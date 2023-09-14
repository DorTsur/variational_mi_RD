from bunch import Bunch
import logging
from collections import OrderedDict
import os
import datetime

logger = logging.getLogger("logger")




def GetConfig(args):
    config = {
        'cuda_visible': 5,  # idx 1->0, idx 3->5, idx 0->2

        'experiment': 'mnist_vae',
        # for setting the model,trainer,data params, options (AWGN, rd_gauss, MINE_NDT_discrete, mnist, gauss_loader, mnist_vae, rdp_gauss, discrete_alt)
        'model': 'MNIST_VAE_BM',  # (MINE, MINE_NDT, MINE_NDT_discrete, MNIST_VAE, MNIST_VAE_BM)
        'trainer': 'Both_Loader',  # (MINE, MINE_NDT, MINE_NDT_discrete, Both_Loader)

        # 'experiment': 'rd_gauss',  # for setting the model,trainer,data params, options (AWGN, rd_gauss)
        # 'model': 'MINE',  # (MINE, MINE_NDT)
        # 'trainer': 'MINE',  # (MINE, MINE_NDT)
        'data': 'MNIST',  # (MINE_awgn, GaussianRD, MINE_NDT_discrete, MNIST, GaussianLoader)

        'x_dim': 1,
        'y_dim': 1,

        'u_dim': 1,

        'u_std': 0.1,
        'u_distribution': 'gaussian',

        'smile_tau': 5.0,

        'D': 0.001,
        'P': 0.0001,
        'regularize': 1,
        'gamma_d': 1000,
        'gamma_discrete': 20,
        'max_gamma': 1000000,

        'gaussian_r': 0.3,

        'batch_size': 2000,  # used 1000 for synthetic and 128 for images
        'num_iter': 80000,
        'eval_batches': 10,
        'lr': 5e-4,

        'critic_hidden_dim': 256,
        'ndt_hidden_dim': 256,
        'critic_embed_dim': 32,
        'critic_layers': 2,
        'critic_activation': 'elu',
        'gsm_temperature': 1.0,
        'eval_freq': 2500,

        'kl_loss': 'dv',  # (smile, infonce, js, dv, tuba, nwj, ent_decomp, regularized_dv, FLO, club)

        'using_wandb': 1,
        'wandb_project_name': 'debug_mnist',

        'clip_grads': 1,
        'grad_clip_val': 0.01,
        'xy_dim': 1,

        'num_epochs': 50,
        'image_batch_size': 64,
        'save_epoch': 15,
        'num_samples': 150000,  # len of synthetic dataloader

        'gamma_p': 20.0,
        'perception': 1,
        'perception_type': 'W2',  #'W2' for wasserstein2 perception

        'alphabet_size': 3,
        'bern_p': 0.5,
        'increase_gamma': 1,

        'noise_encoder': 1,
        'quantize_alphabet': 6,   # quantize_alphabet=0 is means we don't quantize
        'latent_dim': 6,
        'gamma_cap': 100000000,
        'quantizer_centers_reg': 0,
        'out_vae_latent': 1

    }

    config = Config(config)

    # Add args values to the config attributes
    for key in sorted(vars(args)):
        val = getattr(args, key)
        if val is not None:
            setattr(config, key, val)


    # Set some preset params wrt experiment:
    if hasattr(config, 'experiment'):
        if config.experiment == 'awgn':
            config.data, config.model, config.trainer = 'MINE_awgn', 'MINE', 'MINE'
        elif config.experiment == 'rd_gauss':
            config.data, config.model, config.trainer = 'GaussianRD', 'MINE_NDT', 'MINE_NDT'
        elif config.experiment == 'rdp_gauss':
            config.data, config.model, config.trainer, config.x_dim, config.u_dim, config.perception = 'GaussianRDP', 'MINE_NDT', 'MINE_NDT', 1, 1, 1
        elif config.experiment == 'gauss_loader':
            config.data, config.model, config.trainer = 'GaussianLoader', 'MINE_NDT', 'Both_Loader'
        elif config.experiment == 'mnist':
            config.data, config.model, config.trainer, config.batch_size = 'MNIST', 'MNIST', 'Both_Loader', 128
        elif config.experiment == 'mnist_vae':
            config.data, config.trainer, config.batch_size, config.x_dim = 'MNIST', 'Both_Loader', 128, 784
        elif config.experiment == 'discrete_alt':
            config.data, config.model, config.trainer = 'Discrete', 'DiscreteAlt', 'MINE_NDT_DiscreteAlt'

    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    config.dir = f'results/{config.data}/B_{config.batch_size}_{config.model}_{date_string}'

    if not os.path.exists(config.dir):
        os.makedirs(config.dir)


    # config.print()
    return config


class Config(Bunch):
    """ class for handling dictionary as class attributes """
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
    def print(self):
        line_width = 132
        line = "-" * line_width
        logger.info(line + "\n" +
              "| {:^35s} | {:^90} |\n".format('Feature', 'Value') +
              "=" * line_width)
        for key, val in sorted(self.items(), key= lambda x: x[0]):
            if isinstance(val, OrderedDict):
                raise NotImplementedError("Nested configs are not implemented")
            else:
                logger.info("| {:35s} | {:90} |\n".format(key, str(val)) + line)
        logger.info("\n")




# defualt dic before playing with it
    # config = {
    #     'cuda_visible':         1,
    #
    #     'experiment':           'rd_gauss',  # for setting the model,trainer,data params, options (AWGN, rd_gauss)
    #     'model':                'MINE',   # (MINE, MINE_NDT)
    #     'trainer':              'MINE',   # (MINE, MINE_NDT)
    #     'data':                 'MINE_awgn',  # (MINE_awgn, GaussianRD)
    #
    #     'x_dim':                10,
    #     'y_dim':                10,
    #
    #     'u_dim':                10,
    #     'u_std':                1.0,
    #     'u_distribution':       'gaussian',
    #
    #     'smile_tau':            5.0,
    #
    #     'D':                    0.05,
    #     'gaussian_r':           0.3,
    #
    #     'batch_size':           500,
    #     'num_iter':             50000,
    #     'eval_batches':         10,
    #     'lr':                   5e-4,
    #
    #
    #     'critic_hidden_dim':    256,
    #     'critic_embed_dim':     32,
    #     'critic_layers':        2,
    #     'critic_activation':    'relu',
    #
    #     'kl_loss':              'smile',  # (smile, infonce, js, dv, tuba, nwj, ent_decomp)
    #
    #     'using_wandb':          1
    #
    # }

    # In case I want to implement json reading:
    # config = read_json_to_dict(json_file)

    # Turn into Bunch object