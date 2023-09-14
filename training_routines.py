import torch
from utilities import GaussianRateDistortion, GaussianRateDistortionPerception
# from simplified.simplified_model import *
from estimator_fns import estimate_mutual_information, club_loss, log_loss
import os
import numpy as np
import torch.optim as optim
import wandb
import sys
from dataloader import PreprocessBatch
import matplotlib.pyplot as plt
from visualizers import Visualizer
from estimator_fns import club_loss
import matplotlib.pyplot as plt
from utilities import Wasserstein1D
from utilities import hamming_rd, hamming_rdp
import torch.nn as nn
from torch.nn import functional as F





def GetTrainer(config, model, data_gen):
    if config.trainer == 'MINE':
        trainer = TrainMINE(config, model, data_gen)
    elif config.trainer == 'MINE_NDT':
        trainer = TrainBoth(config, model, data_gen)
    elif config.trainer == 'MINE_NDT_DiscreteAlt':
        trainer = TrainBothDiscrete(config, model, data_gen)
    elif config.trainer == 'Both_Loader':
        trainer = TrainWithLoader(config, model, data_gen)
    elif config.trainer == 'MINE_NDT_discrete':
        trainer = TrainDiscreteGSM(config, model, data_gen)
    # elif config.trainer == 'Images':
    #     trainer = TrainImages(config, model, data_gen)
    else:
        raise ValueError("'{}' is an invalid experiment name")
    return trainer



class TrainMINE(object):
    def __init__(self, config, model, data_gen):
        self.config = config
        self.model = model
        self.data_gen = data_gen

        # define optimizer:
        self.opt = {}
        if config.kl_loss == 'ent_decomp':
            self.opt['kl'] = {}
            self.opt['kl']['y'] = optim.Adam(self.model['kl_y'].parameters(), lr=config.lr)
            self.opt['kl']['xy'] = optim.Adam(self.model['kl_y'].parameters(), lr=config.lr)
        elif config.kl_loss == 'FLO':
            self.opt['kl'] = optim.Adam(params=[*self.model['kl']['g'].parameters(), *self.model['kl']['u'].parameters()]
                                        , lr=config.lr)
        else:
            self.opt['kl'] = optim.Adam(self.model['kl'].parameters(), lr=config.lr)

        # calculate expected solution (if theoretically known):
        if config.data == 'MINE_awgn':
            self.expected_sol = 0.5 * np.log(2) * config.x_dim
            print(f'Expected MI for dim {config.x_dim} is {self.expected_sol} [nats]')
        else:
            self.expected_sol = None

        self.using_wandb = False if not hasattr(config, 'using_wandb') else True

        if self.using_wandb:
            wandb.log({'D_eval': config.D})


    def train(self):
        estimates = []
        step = self.config.num_iter // 1000 if self.config.num_iter > 1000 else 1

        for i in range(self.config.num_iter):
            # training_model = 'kl'
            mi = self.train_step()
            mi = mi.detach().cpu().numpy()
            if (i % 25 == 0) and self.expected_sol is not None:
                print(f'Step {i}, estimated mi value {mi}, distance from expected: {mi - self.expected_sol}')
            if not (i % step):
                estimates.append(mi)


        mi = self.eval_step()
        mi = mi.detach().cpu().numpy()
        estimates.append(mi)
        print(f'Final step, estimated mi value {mi}, distance from expected: {mi - self.expected_sol}')


        if not os.path.exists('../results'):
            os.makedirs('results')

        estimates.append(self.expected_sol)
        results = np.array(estimates)
        if self.config.kl_loss != 'smile':
            np.save(f'results/result_kl_{self.config.kl_loss}_dim_{self.config.x_dim}_clip_{self.config.clip}', results)
        else:
            np.save(f'results/result_kl_{self.config.kl_loss}_dim_{self.config.x_dim}', results)

        return estimates


    #######
    def train_step(self):
        if isinstance(self.opt['kl'], dict):
            for key in self.opt['kl']:
                self.opt['kl'][key].zero_grad()
        else:
            self.opt['kl'].zero_grad()

        x, y = self.data_gen.sample()
        x, y = x.cuda(), y.cuda()

        if self.config.kl_loss == 'ent_decomp':
            Dy = estimate_mutual_information(estimator='dy', x=None, y=y, critic_fn=self.model['kl']['y'])
            loss_y = -Dy
            loss_y.backward()
            self.opt['kl']['y'].step()

            Dxy = estimate_mutual_information(estimator='dxy', x=x, y=y, critic_fn=self.model['kl']['xy'])
            loss_xy = -Dxy
            loss_xy.backward()
            self.opt['kl']['xy'].step()

            mi = loss_y - loss_xy

        else:
            clip = None if not(hasattr(self.config, 'smile_tau')) else self.config.smile_tau
            # if self.config.kl_loss == 'smile':
            #     clip = self.config.smile_tau
            # else:
            #     clip = None
            mi = estimate_mutual_information(self.config.kl_loss, x, y, self.model['kl'], clip=clip)
            loss = -mi
            loss.backward()
            self.opt['kl'].step()

        return mi


    def eval_step(self):
        eval = []

        for _ in range(10):
            x, y = self.data_gen.sample()
            x, y = x.cuda(), y.cuda()

            if self.config.kl_loss == 'ent_decomp':
                Dy = estimate_mutual_information(estimator='dy', x=x, y=y, critic_fn=self.model['kl']['y'])
                Dxy = estimate_mutual_information(estimator='dxy', x=None, y=y, critic_fn=self.model['kl']['xy'])
                mi = Dxy-Dy

            else:
                mi = estimate_mutual_information(self.config.kl_loss, x, y, self.model['kl'])

            eval.append(mi)
        return sum(eval)/len(eval)




##################################################################################################################
class TrainBoth(TrainMINE):
    def __init__(self, config, model, data_gen):
        super().__init__(config, model, data_gen)
        self.gamma_p = self.config.gamma_p

        self.opt['ndt'] = optim.Adam(self.model['ndt'].parameters(), lr=config.lr/2)
        if self.config.data == 'GaussianRD':
            self.expected_sol = GaussianRateDistortion(D=config.D, sigmas=self.data_gen.sigmas)
            print(f'Expected rate for distortion {config.D} is {self.expected_sol} [nats]')
        if self.config.data == 'GaussianRDP':
            self.expected_sol = GaussianRateDistortionPerception(D=config.D, P=config.P)
            print(f'Expected rate for distortion {config.D} and perception {config.P} is {self.expected_sol} [nats]')

    def train(self):
        self.estimates = []
        step = self.config.num_iter // 1000 if self.config.num_iter > 1000 else 1

        for i in range(self.config.num_iter):
            if i % self.config.eval_freq == 0:
                mi_eval = self.eval_step(mode='final')
                mi_eval = mi_eval.detach().cpu().numpy()
                self.estimates.append(mi_eval)
                print(f'Eval step, estimated mi value {mi_eval}, distance from expected: {mi_eval - self.expected_sol}')
                if self.using_wandb:
                    wandb.log({'mi_eval': mi_eval})
                    if self.config.data in ('GaussianRD','GaussianRDP'):
                        wandb.log({'est-expected_eval': mi_eval - self.expected_sol})

            if not (i % 3) and i > 0:
                training_model = 'ndt'
            else:
                training_model = 'kl'

            if i % 100 == 0 and self.config.increase_gamma:
                # self.gamma_p = min(self.gamma_p * 2, float(1e4))
                self.gamma_p = min(self.gamma_p * 2, self.config.max_gamma)



            mi = self.train_step(training_model)
            mi = mi.detach().cpu().numpy()

            if i%25 == 0:
                print(f'{i}th Iter, MI: {mi:.4f}, Distance: {mi - self.expected_sol:.4f}')
            if not (i % step) and training_model == 'kl':
                self.estimates.append(mi)
                if self.using_wandb:
                    wandb.log({'mi_train': mi})

            if np.isnan(mi):
                mi = self.eval_step(mode='final')
                mi = mi.detach().cpu().numpy()
                print(f'Final step, estimated mi value {mi}, distance from expected: {mi - self.expected_sol}')
                if self.using_wandb:
                    wandb.log({'mi_final': mi})
                sys.exit("Simulation stopped: mi is NaN")

            self.tail_avg = np.mean(self.estimates[-100:])
            if self.using_wandb:
                wandb.log({'mi_tail_avg': self.tail_avg})
                wandb.log({'iteration': i})
                if self.config.data in ('GaussianRD', 'GaussianRDP'):
                    wandb.log({'est-expected': self.tail_avg-self.expected_sol})


        mi = self.eval_step(mode='final')
        mi = mi.detach().cpu().numpy()
        self.estimates.append(mi)
        print(f'Final step, estimated mi value {mi}, distance from expected: {mi - self.expected_sol}')
        if self.using_wandb:
            wandb.log({'mi_final': mi})
            if self.config.data in ('GaussianRD', 'GaussianRDP'):
                wandb.log({'est-expected_final': mi - self.expected_sol})


        # if not os.path.exists('../results'):
        #     os.makedirs('results')

        self.estimates.append(self.expected_sol)
        # results = np.array(self.estimates)

        # if self.config.kl_loss == 'smile':
        #     np.save(f'results/result_kl_{self.config.kl_loss}_dim_{self.config.x_dim}_clip_{self.config.clip}_{self.config.model}', results)
        # else:
        #     np.save(f'results/result_kl_{self.config.kl_loss}_dim_{self.config.x_dim}_{self.config.model}', results)

        return self.estimates

    def train_step(self, training_model):
        if isinstance(self.opt['kl'], dict):
            for key in self.opt['kl']:
                self.opt['kl'][key].zero_grad()
        else:
            self.opt['kl'].zero_grad()

        self.opt['ndt'].zero_grad()

        x = self.data_gen.sample_x()
        u = self.data_gen.sample_u()

        x = x.to(u.dtype)
        x, u = x.cuda(), u.cuda()
        y = self.model['ndt'](x, u)

        if training_model == 'kl':
            if self.config.kl_loss == 'ent_decomp':
                Dy = estimate_mutual_information(estimator='dy', x=None, y=y, critic_fn=self.model['kl']['y'])
                loss_y = -Dy
                loss_y.backward()
                self.opt['kl']['y'].step()

                Dxy = estimate_mutual_information(estimator='dxy', x=x, y=y, critic_fn=self.model['kl']['xy'])
                loss_xy = -Dxy
                loss_xy.backward()
                self.opt['kl']['xy'].step()

                mi = loss_y-loss_xy

            else:
                mi = estimate_mutual_information(self.config.kl_loss, x, y, self.model['kl'])
                loss = -mi
                loss.backward()

                if self.config.clip_grads:
                    if self.config.kl_loss == 'FLO':
                        torch.nn.utils.clip_grad_norm_(
                            [*self.model['kl']['g'].parameters(), *self.model['kl']['u'].parameters()],
                            self.config.grad_clip_val)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model['kl'].parameters(), self.config.grad_clip_val)

                self.opt['kl'].step()
        else:
            if self.config.kl_loss == 'ent_decomp':
                mi = None
                pass
                # mi = ...
                # blah, returns mi
            else:
                mi = estimate_mutual_information(self.config.kl_loss, x, y, self.model['kl'])
                if self.config.perception:
                    mi = mi + self.gamma_p * self.perception_reg(x, y)


            mi.backward()

            if self.config.clip_grads:
                torch.nn.utils.clip_grad_norm_(self.model['ndt'].parameters(), self.config.grad_clip_val)

            self.opt['ndt'].step()
        return mi

    def perception_reg(self, x, y):
        if not self.config.perception:
            return 0
        if self.config.perception_type == 'W2':
            return torch.square(Wasserstein1D(x, y, p=2)-self.config.P)
        if self.config.perception_type == 'W2_gauss':
            pass  # TD if needed

    def eval_step(self, mode='final'):
        eval = []
        batches = self.config.eval_batches if mode == 'final' else 2
        for _ in range(batches):
            # sample from decay sigmas model:
            x = self.data_gen.sample_x('eval')
            u = self.data_gen.sample_u('eval')

            # pass through ndt
            x = x.to(u.dtype)
            x, u = x.cuda(), u.cuda()
            y = self.model['ndt'](x, u)

            if self.config.kl_loss == 'ent_decomp':
                Dy = estimate_mutual_information(estimator='dy', x=x, y=y, critic_fn=self.model['kl']['y'])
                Dxy = estimate_mutual_information(estimator='dxy', x=None, y=y, critic_fn=self.model['kl']['xy'])
                mi = Dxy-Dy

            else:
                mi = estimate_mutual_information(self.config.kl_loss, x, y, self.model['kl'])

            eval.append(mi)

        torch.cuda.empty_cache()

        return sum(eval)/len(eval)

    def process_results(self, save=True):
        if not os.path.exists('results'):
            os.makedirs('results')

        # save estimates to numpy file in results directory
        if self.config.kl_loss == 'smile':
            filename = f'results/{self.config.experiment}_batch_{self.config.batch_size}_activation_' \
                       f'{self.config.critic_activation}_D_{self.config.D}_{self.config.kl_loss}_tau_{self.config.tau_smile}.npy'
        else:
            filename = f'results/{self.config.experiment}_batch_{self.config.batch_size}_activation_' \
                       f'{self.config.critic_activation}_D_{self.config.D}_{self.config.kl_loss}.npy'
        np.save(filename, self.estimates)


class TrainBothDiscrete(TrainMINE):
    def __init__(self, config, model, data_gen):
        super().__init__(config, model, data_gen)

        self.binary_rd_data()
        self.opt['ndt'] = optim.Adam(self.model['ndt'].parameters(), lr=config.lr/2)
        if self.config.data == 'Discrete':
            if config.perception and config.alphabet_size == 2:
                self.expected_sol = hamming_rdp(config.D, config.P, config.bern_p)
                print(f'Expected rate for distortion {config.D} and perception {config.P} is {self.expected_sol} [nats]')
            else:
                self.expected_sol = hamming_rd(config.alphabet_size, config.D, config.bern_p)
                print(f'Expected rate for distortion {config.D} is {self.expected_sol} [nats]')

        self.gamma_p = self.config.gamma_discrete

    def train(self):
        self.estimates = []
        step = self.config.num_iter // 1000 if self.config.num_iter > 1000 else 1

        for i in range(self.config.num_iter):
            if i % self.config.eval_freq == 0:
                mi_eval = self.eval_step(mode='final')
                mi_eval = mi_eval.detach().cpu().numpy()
                self.estimates.append(mi_eval)
                print(f'Eval step, estimated mi value {mi_eval}, distance from expected: {mi_eval - self.expected_sol}')
                if self.using_wandb:
                    wandb.log({'mi_eval': mi_eval})
                    if self.config.data == 'Discrete':
                        wandb.log({'est-expected_eval': mi_eval - self.expected_sol})

            if not (i % 3) and i > 0:
                training_model = 'ndt'
            else:
                training_model = 'kl'


            mi = self.train_step(training_model)
            mi = mi.detach().cpu().numpy()

            if training_model == 'kl':
                if i % 100 == 0:
                    d = self.calc_d_hamming()
                    p = self.calc_p_tv()
                    if self.config.alphabet_size >= 2:
                        self.gamma_p = min(self.gamma_p*2, float(1e4))
                    print(f'{i}th Iter, MI: {mi:.4f}, Distance: {mi - self.expected_sol:.4f}, est_D: {d:.3f}, est_p: {p:.3f}')
                if not (i % step) and training_model == 'kl':
                    self.estimates.append(mi)
                    if self.using_wandb:
                        wandb.log({'mi_train': mi})

                if np.isnan(mi):
                    mi = self.eval_step(mode='final')
                    mi = mi.detach().cpu().numpy()
                    print(f'Final step, estimated mi value {mi}, distance from expected: {mi - self.expected_sol}')
                    if self.using_wandb:
                        wandb.log({'mi_final': mi})
                    sys.exit("Simulation stopped: mi is NaN")

                self.tail_avg = np.mean(self.estimates[-100:])
                if self.using_wandb:
                    wandb.log({'mi_tail_avg': self.tail_avg})
                    wandb.log({'iteration': i})
                    if self.config.data == 'Discrete':
                        wandb.log({'est-expected': self.tail_avg-self.expected_sol})


        mi = self.eval_step(mode='final')
        mi = mi.detach().cpu().numpy()
        self.estimates.append(mi)
        print(f'Final step, estimated mi value {mi}, distance from expected: {mi - self.expected_sol}')
        if self.using_wandb:
            wandb.log({'mi_final': mi})
            if self.config.data == 'Discrete':
                wandb.log({'est-expected_final': mi - self.expected_sol})


        # if not os.path.exists('../results'):
        #     os.makedirs('results')

        self.estimates.append(self.expected_sol)
        # results = np.array(self.estimates)

        # if self.config.kl_loss == 'smile':
        #     np.save(f'results/result_kl_{self.config.kl_loss}_dim_{self.config.x_dim}_clip_{self.config.clip}_{self.config.model}', results)
        # else:
        #     np.save(f'results/result_kl_{self.config.kl_loss}_dim_{self.config.x_dim}_{self.config.model}', results)

        return self.estimates

    def train_step(self, training_model):
        if isinstance(self.opt['kl'], dict):
            for key in self.opt['kl']:
                self.opt['kl'][key].zero_grad()
        else:
            self.opt['kl'].zero_grad()

        self.opt['ndt'].zero_grad()

        x = self.data_gen.sample_x()
        x = x.cuda()
        py_x = self.model['ndt'](x)
        y = self.model['sampler'](py_x)

        # training_model = 'kl'

        if training_model == 'kl':
            y = nn.functional.one_hot(y, self.config.alphabet_size).to(torch.float32).squeeze(dim=1)

            # x, y = self.binary_rd_data()

            if self.config.kl_loss == 'ent_decomp':
                Dy = estimate_mutual_information(estimator='dy', x=None, y=y, critic_fn=self.model['kl']['y'])
                loss_y = -Dy
                loss_y.backward()
                self.opt['kl']['y'].step()

                Dxy = estimate_mutual_information(estimator='dxy', x=x, y=y, critic_fn=self.model['kl']['xy'])
                loss_xy = -Dxy
                loss_xy.backward()
                self.opt['kl']['xy'].step()

                mi = loss_y-loss_xy

            else:
                mi = estimate_mutual_information(self.config.kl_loss, x, y, self.model['kl'])
                loss = -mi
                loss.backward()

                if self.config.clip_grads:
                    if self.config.kl_loss == 'FLO':
                        torch.nn.utils.clip_grad_norm_(
                            [*self.model['kl']['g'].parameters(), *self.model['kl']['u'].parameters()],
                            self.config.grad_clip_val)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model['kl'].parameters(), self.config.grad_clip_val)

                self.opt['kl'].step()
        else:
            if self.config.kl_loss == 'ent_decomp':
                mi = None
                pass
                # mi = ...
                # blah, returns mi
            else:
                y_oh = nn.functional.one_hot(y, self.config.alphabet_size).to(torch.float32).squeeze(dim=1)
                mi = estimate_mutual_information('alt_loss', py_x, y_oh, self.model['kl'], y_int=y)
                mi = mi + self.distortion_reg(x, py_x)
                if self.config.perception:
                    mi = mi + self.perception_reg(x, py_x)
                # if self.config.perception:
                #     mi = mi + self.config.gamma_p * self.perception_reg(x, y)


            mi.backward()

            if self.config.clip_grads:
                torch.nn.utils.clip_grad_norm_(self.model['ndt'].parameters(), self.config.grad_clip_val)

            self.opt['ndt'].step()
        return mi

    def distortion_reg(self, x, p):
        x_hamming = (1.0 - x)
        d = torch.mean(torch.sum(p * x_hamming, dim=-1))
        reg = self.gamma_p * (d - self.config.D) ** 2
        return reg

    def perception_reg(self, x, p):
        """
        calcaulte the total variation distance perception regularization
        """
        if self.config.alphabet_size == 2:
            px_0, px_1 = x.mean(dim=0)
            py_0, py_1 = p.mean(dim=0)

            TV = 0.5*( (px_0-py_0).abs() + (px_1-py_1).abs() )

        else:
            px = x.mean(dim=0)
            py = p.mean(dim=0)
            TV = 0.5*((px-py).abs().sum())

        return self.gamma_p*(TV-self.config.P).square()

    def eval_step(self, mode='final'):
        eval = []
        batches = self.config.eval_batches if mode == 'final' else 2
        for _ in range(batches):
            # sample from decay sigmas model:
            x = self.data_gen.sample_x('eval')

            # pass through ndt
            x = x.cuda()
            py_x = self.model['ndt'](x)
            y = self.model['sampler'](py_x)
            y = nn.functional.one_hot(y, self.config.alphabet_size).to(torch.float32).squeeze(dim=1)

            mi = estimate_mutual_information(self.config.kl_loss, x, y, self.model['kl'])

            eval.append(mi)

        torch.cuda.empty_cache()

        return sum(eval)/len(eval)

    def process_results(self, save=True):
        if not os.path.exists('results'):
            os.makedirs('results')

        # save estimates to numpy file in results directory
        if self.config.kl_loss == 'smile':
            filename = f'results/{self.config.experiment}_batch_{self.config.batch_size}_activation_' \
                       f'{self.config.critic_activation}_D_{self.config.D}_{self.config.kl_loss}_tau_{self.config.tau_smile}.npy'
        else:
            filename = f'results/{self.config.experiment}_batch_{self.config.batch_size}_activation_' \
                       f'{self.config.critic_activation}_D_{self.config.D}_{self.config.kl_loss}.npy'
        np.save(filename, self.estimates)

    def binary_rd_data(self):
        # RD data
        # Calculate py
        # py_val = max(0, (self.config.bern_p - self.config.D) / (1.0 - 2 * self.config.D))
        # py = torch.tensor([[1 - py_val, py_val]]*self.config.batch_size)
        #
        # # Sample y using the calculated py
        # # y_logits = torch.log(py)
        # y = torch.multinomial(py, num_samples=1)
        # # y = y.view(self.config.batch_size, 1)
        #
        # # Sample z
        # pz = torch.tensor([[1 - self.config.D, self.config.D]]*self.config.batch_size)
        # z = torch.multinomial(pz, num_samples=1)
        # # z = z.view(self.config.batch_size, 1)
        #
        # # Calculate x
        # x = (y + z) % 2

        # bsc data:
        px = torch.tensor([[0.5, 0.5]] * self.config.batch_size)
        pz = torch.tensor([[0.1, 0.9]] * self.config.batch_size)

        x = torch.multinomial(px, num_samples=1)
        z = torch.multinomial(pz, num_samples=1)

        y = (x + z) % 2

        y = nn.functional.one_hot(y, self.config.alphabet_size).to(torch.float32).squeeze(dim=1).float().cuda()
        x = nn.functional.one_hot(x, self.config.alphabet_size).to(torch.float32).squeeze(dim=1).float().cuda()
        return x, y

    def calc_d_hamming(self):
        x = self.data_gen.sample_x()
        x = x.cuda()
        py_x = self.model['ndt'](x)
        x_hamming = (1.0 - x)
        d = torch.mean(torch.sum(py_x * x_hamming, dim=-1))
        return d.detach().cpu().numpy()

    def calc_p_tv(self):
        x = self.data_gen.sample_x()
        x = x.cuda()
        py_x = self.model['ndt'](x)
        if self.config.alphabet_size==2:
            px_0, px_1 = x.mean(dim=0)
            py_0, py_1 = py_x.mean(dim=0)

            TV = 0.5 * ((px_0 - py_0).abs() + (px_1 - py_1).abs())
        else:
            px = x.mean(dim=0)
            py = py_x.mean(dim=0)
            TV = 0.5 * ((px - py).abs().sum())
        return TV.detach().cpu().numpy()




##################################################################################################################
class TrainWithLoader(TrainMINE):
    def __init__(self, config, model, dataloader):
        super().__init__(config, model, dataloader)
        self.opt['ndt'] = optim.Adam(self.model['ndt'].parameters(), lr=config.lr/2)
        self.dataloader = dataloader
        self.visualizer = Visualizer(config)
        self.gamma = config.gamma_d
        self.distortion = 'mse'

    def train(self):
        # Simple implementation of joint training - not alternating
        if self.config.kl_loss == 'club':
            self.train_club()
            return
        mi_list = []
        for epoch in range(self.config.num_epochs):
            mi_epoch = []
            distortion_epoch = []
            if self.config.increase_gamma:
                self.gamma = min(self.config.gamma_cap, 10*self.gamma)
            for i, (x, _) in enumerate(self.dataloader, 0):
                # KL step:
                for opt in self.opt.values():
                    opt.zero_grad()

                x = x.cuda()
                if self.config.data == 'MNIST':
                    # if self.config.experiment == 'mnist_vae':
                        # x = x.reshape(-1, 784)  # flatten x
                    y = self.model['ndt'](x)
                    x = x.reshape(-1, 784)
                    y = y.reshape(-1, 784)
                else:
                    if self.config.u_distribution == "gaussian":
                        u = self.config.u_std * torch.randn(size=x.shape).cuda()
                    else:
                        u = torch.rand(size=x.shape).cuda()

                    x = x.to(u.dtype)
                    y = self.model['ndt'](x, u)


                mi = estimate_mutual_information(self.config.kl_loss, x, y, self.model['kl'], 'q_loss')
                loss = -mi  # -mi for maximization
                loss.backward()

                if self.config.clip_grads:
                    torch.nn.utils.clip_grad_norm_(self.model['kl'].parameters(), self.config.grad_clip_val)
                self.opt['kl'].step()

                # mu, logvar = self.model['kl'](x)
                # club_val = club_loss((mu, logvar, y))
                # print(f'club_val = {club_val}')

                # NDT step:
                for opt in self.opt.values():
                    opt.zero_grad()

                if self.config.data == 'MNIST':
                    y = self.model['ndt'](x)
                    y = y.reshape(-1, 784)
                else:
                    x = x.to(u.dtype)
                    y = self.model['ndt'](x, u)

                mi = estimate_mutual_information(self.config.kl_loss, x, y, self.model['kl'])
                if self.config.regularize:
                    loss = mi + self.gamma*self.distortion_reg(x, y)
                else:
                    loss = mi

                loss = loss + self.config.gamma_p*self.perception_reg(x,y)
                loss.backward()

                if self.config.clip_grads:
                    torch.nn.utils.clip_grad_norm_(self.model['ndt'].parameters(), self.config.grad_clip_val)
                self.opt['ndt'].step()

                distortion = self.calc_distortion(x,y)


                mi_epoch.append(mi.detach().cpu().numpy())
                distortion_epoch.append(distortion.numpy())
                if i != 0 and i % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{self.config.num_epochs}], Iteration [{i}/{len(self.dataloader)}]')

            print(self.model['ndt'].quantizer.centers)
            mi_epoch = np.mean(mi_epoch)
            distortion_epoch = np.mean(distortion_epoch)
            print(f'Epoch {epoch+1}, Estimated MI {mi_epoch:.3f} [nats], Distortion {distortion_epoch:.3f}')
            mi_list.append(mi_epoch)

            if epoch % self.config.save_epoch == 0 and self.config.data == 'MNIST':
                self.visualizer.visualize(epoch=epoch, dataloader=self.dataloader, model=self.model['ndt'])

            if self.config.using_wandb:
                wandb.log({'mi_train': mi_epoch})
                wandb.log({'d_train': distortion_epoch})
                wandb.log({'Epoch': epoch})


        # self.plot_result(mi_list)

    def train_club(self):
        """
        Training routine that combines the CLUB and a VAE model.
        """
        mi_list = []
        for epoch in range(self.config.num_epochs):
            mi_epoch = []
            distortion_epoch = []
            if self.config.increase_gamma:
                self.gamma = min(self.config.gamma_cap, 10*self.gamma)

            for i, (x, _) in enumerate(self.dataloader, 0):
                # CLUB step:
                for opt in self.opt.values():
                    opt.zero_grad()

                x = x.cuda()
                y, mu, logvar = self.model['ndt'](x)
                x = x.reshape(-1, 784)
                y = y.reshape(-1, 784)

                loss = log_loss((mu, logvar, y))
                # verify we don't need a '-'
                loss.backward()

                if self.config.clip_grads:
                    torch.nn.utils.clip_grad_norm_(self.model['kl'].parameters(), self.config.grad_clip_val)
                self.opt['kl'].step()

                # VAE step:
                for opt in self.opt.values():
                    opt.zero_grad()

                for opt in self.opt.values():
                    opt.zero_grad()

                x = x.cuda()
                y, mu, logvar = self.model['ndt'](x)
                x = x.reshape(-1, 784)
                y = y.reshape(-1, 784)

                loss = club_loss((mu, logvar, y))
                # verify we don't need a '-'
                distortion_reg = self.gamma*self.distortion_reg(x, y)
                loss += distortion_reg

                if self.config.quantizer_centers_reg and self.config.quantize_alphabet > 0:
                    loss += 0.1*torch.sum(self.model['ndt'].quantizer.centers ** 2)

                if self.config.perception:
                    loss += self.perception_reg(x, y)

                loss.backward()

                if self.config.clip_grads:
                    torch.nn.utils.clip_grad_norm_(self.model['ndt'].parameters(), self.config.grad_clip_val)
                self.opt['ndt'].step()

                distortion = self.calc_distortion(x, y)


                mi_epoch.append(loss.detach().cpu().numpy())
                distortion_epoch.append(distortion.numpy())
                if i != 0 and i % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{self.config.num_epochs}], Iteration [{i}/{len(self.dataloader)}]')

            mi_epoch = np.mean(mi_epoch)
            distortion_epoch = np.mean(distortion_epoch)
            print(f'Epoch {epoch+1}, Estimated MI {mi_epoch:.3f} [nats], Distortion {distortion_epoch:.3f}')
            mi_list.append(mi_epoch)



            if epoch % self.config.save_epoch == 0 and self.config.data == 'MNIST':
                self.visualizer.visualize(epoch=epoch, dataloader=self.dataloader, model=self.model['ndt'])

            if self.config.using_wandb:
                wandb.log({'mi_train': mi_epoch})
                wandb.log({'d_train': distortion_epoch})
                wandb.log({'Epoch': epoch})

        # self.plot_result(mi_list)

    def plot_result(self, data):
        # Plot the training losses.
        plt.figure(figsize=(10, 5))
        plt.title("MI vs. Iteration")
        plt.plot(data, label="Estaimted MI")
        plt.xlabel("iterations")
        plt.ylabel("MI [nats]")
        plt.legend()
        plt.savefig(self.config.figDir + "/MI Loss Curve")

    def distortion_reg(self, x, y):
        if self.config.data == 'MNIST':
            return torch.square(nn.MSELoss()(x,y) - self.config.D)
        if self.distortion == 'mse':
            return torch.square(torch.mean(torch.square(torch.linalg.norm(x - y, dim=-1))) - self.config.D)

    def perception_reg(self, x, y):
        if not self.config.perception:
            return 0.0
        if self.config.perception_type == 'W2' and self.config.x_dim == 1:
            return Wasserstein1D(x, y, p=2)
        if self.config.perception_type == 'W2_gauss':
            pass  # TD if needed
        return 0.0

    def eval_step(self, mode='final'):
        eval = []
        batches = self.config.eval_batches if mode == 'final' else 2
        for _ in range(batches):
            # sample from decay sigmas model:
            x = self.data_gen.sample_x()
            u = self.data_gen.sample_u()

            # pass through ndt
            x = x.to(u.dtype)
            x, u = x.cuda(), u.cuda()
            y = self.model['ndt'](x, u)

            if self.config.kl_loss == 'ent_decomp':
                Dy = estimate_mutual_information(estimator='dy', x=x, y=y, critic_fn=self.model['kl']['y'])
                Dxy = estimate_mutual_information(estimator='dxy', x=None, y=y, critic_fn=self.model['kl']['xy'])
                mi = Dxy-Dy

            else:
                mi = estimate_mutual_information(self.config.kl_loss, x, y, self.model['kl'])

            eval.append(mi)

        torch.cuda.empty_cache()

        if self.config.using_wandb:
            entropy, ub = self.model['ndt'].entropy_coded(x)
            wandb.log({'rate_ub': ub})
            wandb.log({'huffman_entropy': entropy})
            print(f'rate {ub} and entropy {entropy}')

        return sum(eval)/len(eval)

    def process_results(self, save=True):
        if not os.path.exists('results'):
            os.makedirs('results')

        # save estimates to numpy file in results directory
        if self.config.kl_loss == 'smile':
            filename = f'results/{self.config.experiment}_batch_{self.config.batch_size}_activation_' \
                       f'{self.config.critic_activation}_D_{self.config.D}_{self.config.kl_loss}_tau_{self.config.tau_smile}.npy'
        else:
            filename = f'results/{self.config.experiment}_batch_{self.config.batch_size}_activation_' \
                       f'{self.config.critic_activation}_D_{self.config.D}_{self.config.kl_loss}.npy'
        np.save(filename, self.estimates)

    def calc_distortion(self,x,y):
        x = x.detach().cpu()
        y = y.detach().cpu()

        if self.config.data == 'MNIST':
            return nn.MSELoss()(x, y)
        return torch.square(torch.mean(torch.square(torch.linalg.norm(x - y, dim=-1))) - self.config.D)




##################################################################################################################
class TrainDiscreteGSM(TrainMINE):
    def __init__(self, config, model, data_gen):
        super().__init__(config, model, data_gen)

        self.opt['ndt'] = optim.Adam(self.model['ndt'].parameters(), lr=config.lr/2)
        if self.config.data == 'DiscreteHamming':
            # TD
            self.expected_sol = 0
            print(f'Expected rate for distortion {config.D} is {self.expected_sol} [nats]')




    def train(self):
        self.estimates = []
        step = self.config.num_iter // 1000 if self.config.num_iter > 1000 else 1

        for i in range(self.config.num_iter):
            if not (i % 3) and i > 0:
                training_model = 'ndt'
            else:
                training_model = 'kl'


            mi = self.train_step(training_model)
            mi = mi.detach().cpu().numpy()

            if i%25 == 0:
                print(f'{i}th Iter, MI: {mi:.4f}, Distance: {mi - self.expected_sol:.4f}')
            if not (i % step) and training_model == 'kl':
                self.estimates.append(mi)
                if self.using_wandb:
                    wandb.log({'mi_train': mi})

            if np.isnan(mi):
                mi = self.eval_step(mode='final')
                mi = mi.detach().cpu().numpy()
                print(f'Final step, estimated mi value {mi}, distance from expected: {mi - self.expected_sol}')
                if self.using_wandb:
                    wandb.log({'mi_final': mi})
                sys.exit("Simulation stopped: mi is NaN")

            self.tail_avg = np.mean(self.estimates[-100:])
            if self.using_wandb:
                wandb.log({'mi_tail_avg': self.tail_avg})
                wandb.log({'iteration': i})
                if self.config.data == 'DiscreteHamming':
                    wandb.log({'est-expected': self.tail_avg-self.expected_sol})



        mi = self.eval_step(mode='final')
        mi = mi.detach().cpu().numpy()
        self.estimates.append(mi)
        print(f'Final step, estimated mi value {mi}, distance from expected: {mi - self.expected_sol}')
        if self.using_wandb:
            wandb.log({'mi_final': mi})
            if self.config.data == 'DiscreteHamming':
                wandb.log({'est-expected_final': mi - self.expected_sol})

        self.estimates.append(self.expected_sol)

        return self.estimates


    def train_step(self, training_model):
        if isinstance(self.opt['kl'], dict):
            for key in self.opt['kl']:
                self.opt['kl'][key].zero_grad()
        else:
            self.opt['kl'].zero_grad()

        self.opt['ndt'].zero_grad()

        x = self.data_gen.sample_x()

        x = x.cuda()

        y = self.model['ndt'](x)

        if training_model == 'kl':
            mi = estimate_mutual_information(self.config.kl_loss, x, y, self.model['kl'])
            loss = -mi
            loss.backward()

            if self.config.clip_grads:
                torch.nn.utils.clip_grad_norm_(self.model['kl'].parameters(), self.config.grad_clip_val)

            self.opt['kl'].step()
        else:
            mi = estimate_mutual_information(self.config.kl_loss, x, y, self.model['kl'])
            mi.backward()

            if self.config.clip_grads:
                torch.nn.utils.clip_grad_norm_(self.model['ndt'].parameters(), self.config.grad_clip_val)

            self.opt['ndt'].step()
        return mi


    def eval_step(self, mode='final'):
        eval = []
        batches = self.config.eval_batches if mode == 'final' else 2
        for _ in range(batches):
            # sample from decay sigmas model:
            x = self.data_gen.sample_x()

            # pass through ndt
            # x = x.to(u.dtype)
            x = x.cuda()
            y = self.model['ndt'](x)

            mi = estimate_mutual_information(self.config.kl_loss, x, y, self.model['kl'])

            eval.append(mi)

        torch.cuda.empty_cache()

        return sum(eval)/len(eval)


    def process_results(self, save=True):
        if not os.path.exists('results'):
            os.makedirs('results')

        # save estimates to numpy file in results directory
        filename = f'results/{self.config.experiment}_batch_{self.config.batch_size}_activation_' \
                       f'{self.config.critic_activation}_D_{self.config.D}_{self.config.kl_loss}.npy'
        np.save(filename, self.estimates)




##################################################################################################################
# class TrainImages(TrainMINE):
#     def __init__(self, config, model, data_gen):
#         super().__init__(config, model, data_gen)
#
#         self.opt['ndt'] = optim.Adam(self.model['ndt'].parameters(), lr=config.lr/2)
#
#         if config.data == 'GaussianRD':
#             self.expected_sol = GaussianRateDistortion(config.D, data_gen.sigmas)
#             print(f'Expected rate for distortion {config.D} is {self.expected_sol} [nats]')
#             if self.using_wandb:
#                 wandb.log({'expected_sol': self.expected_sol})
#
#
#     def train(self):
#         self.estimates = []
#         step = self.config.num_iter // 1000 if self.config.num_iter > 1000 else 1
#
#         for epoch in self.config.num_epochs:
#             for i, (batch, labels) in enumerate(self.data_gen.train_loader):
#                 if not (i % 3) and i > 0:
#                     training_model = 'ndt'
#                 else:
#                     training_model = 'kl'
#
#                 mi = self.train_step(training_model, batch)
#                 mi = mi.detach().cpu().numpy()
#
#                 if i % 10 == 0:
#                     print(f'Step {i*(epoch+1)}, estimated mi value {mi}')
#
#                 if not (i % step) and training_model == 'kl':
#                     self.estimates.append(mi)
#                     if self.using_wandb:
#                         wandb.log({'mi_train': mi})
#
#                 if np.isnan(mi):
#                     mi = self.eval_step(mode='final')
#                     mi = mi.detach().cpu().numpy()
#                     if self.using_wandb:
#                         wandb.log({'mi_final': mi})
#                     sys.exit("Simulation stopped: mi is NaN")
#
#                 self.tail_avg = np.mean(self.estimates[-100:])
#                 if self.using_wandb:
#                     wandb.log({'mi_tail_avg': self.tail_avg})
#                     wandb.log({'iteration': i})
#
#         return self.estimates
#
#
#     def train_step(self, training_model, batch):
#         if isinstance(self.opt['kl'], dict):
#             for key in self.opt['kl']:
#                 self.opt['kl'][key].zero_grad()
#         else:
#             self.opt['kl'].zero_grad()
#
#         self.opt['ndt'].zero_grad()
#
#         x = batch
#         u = self.data_gen.sample_u(x)
#
#         x = x.to(u.dtype)
#         x, u = x.cuda(), u.cuda()
#         y = self.model['ndt'](x, u)
#
#         if training_model == 'kl':
#             if self.config.kl_loss == 'ent_decomp':
#                 Dy = estimate_mutual_information(estimator='dy', x=None, y=y, critic_fn=self.model['kl']['y'])
#                 loss_y = -Dy
#                 loss_y.backward()
#                 self.opt['kl']['y'].step()
#
#                 Dxy = estimate_mutual_information(estimator='dxy', x=x, y=y, critic_fn=self.model['kl']['xy'])
#                 loss_xy = -Dxy
#                 loss_xy.backward()
#                 self.opt['kl']['xy'].step()
#
#                 mi = loss_y-loss_xy
#
#             else:
#                 mi = estimate_mutual_information(self.config.kl_loss, x, y, self.model['kl'])
#                 loss = -mi
#                 loss.backward()
#
#                 if self.config.clip_grads:
#                     if self.config.kl_loss == 'FLO':
#                         torch.nn.utils.clip_grad_norm_(
#                             [*self.model['kl']['g'].parameters(), *self.model['kl']['u'].parameters()],
#                             self.config.grad_clip_val)
#                     else:
#                         torch.nn.utils.clip_grad_norm_(self.model['kl'].parameters(), self.config.grad_clip_val)
#
#                 self.opt['kl'].step()
#         else:
#             if self.config.kl_loss == 'ent_decomp':
#                 mi = None
#                 pass
#                 # mi = ...
#                 # blah, returns mi
#             else:
#                 mi = estimate_mutual_information(self.config.kl_loss, x, y, self.model['kl'])
#             mi.backward()
#
#             if self.config.clip_grads:
#                 torch.nn.utils.clip_grad_norm_(self.model['ndt'].parameters(), self.config.grad_clip_val)
#
#             self.opt['ndt'].step()
#         return mi
#
#
#     def eval_step(self, mode='final'):
#         eval = []
#         batches = self.config.eval_batches if mode == 'final' else 2
#         for _ in range(batches):
#             # sample from decay sigmas model:
#             x = self.data_gen.sample_x()
#             u = self.data_gen.sample_u()
#
#             # pass through ndt
#             x = x.to(u.dtype)
#             x, u = x.cuda(), u.cuda()
#             y = self.model['ndt'](x, u)
#
#             if self.config.kl_loss == 'ent_decomp':
#                 Dy = estimate_mutual_information(estimator='dy', x=x, y=y, critic_fn=self.model['kl']['y'])
#                 Dxy = estimate_mutual_information(estimator='dxy', x=None, y=y, critic_fn=self.model['kl']['xy'])
#                 mi = Dxy-Dy
#
#             else:
#                 mi = estimate_mutual_information(self.config.kl_loss, x, y, self.model['kl'])
#
#             eval.append(mi)
#
#         torch.cuda.empty_cache()
#
#         return sum(eval)/len(eval)
#
#
#     def process_results(self, save=True):
#         if not os.path.exists('results'):
#             os.makedirs('results')
#
#         # save estimates to numpy file in results directory
#         if self.config.kl_loss == 'smile':
#             filename = f'results/{self.config.experiment}_batch_{self.config.batch_size}_activation_' \
#                        f'{self.config.critic_activation}_D_{self.config.D}_{self.config.kl_loss}_tau_{self.config.tau_smile}.npy'
#         else:
#             filename = f'results/{self.config.experiment}_batch_{self.config.batch_size}_activation_' \
#                        f'{self.config.critic_activation}_D_{self.config.D}_{self.config.kl_loss}.npy'
#         np.save(filename, self.estimates)
