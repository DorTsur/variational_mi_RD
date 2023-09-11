import numpy as np
import torch
from torchvision import datasets, transforms
import torch.nn as nn
from dataloader import GetDataLoader
from torch.nn import functional as F

def GetDataGen(config):
    if config.data == 'MINE_awgn':
        # pass
        data_gen = DataGenAWGN(config)
    elif config.data == 'GaussianRD':
        data_gen = DataGenRDGauss(config)
    elif config.data == 'GaussianRDP':
        data_gen = DataGenRDPGauss(config)
    elif config.data == 'MINE_NDT_discrete':
        data_gen = DataGenRDCategorical(config)
    elif config.data == 'MNIST':
        data_gen = GetDataLoader(config)
    elif config.data == 'GaussianLoader':
        data_gen = GetDataLoader(config)
    elif config.data == 'Discrete':
        data_gen = DataGenRDCategorical(config)
    else:
        raise ValueError("'{}' is an invalid data generator name")
    return data_gen


class DataGenRDGauss(object):
    def __init__(self, config):
        self.config = config
        self.batch_size = config.batch_size
        self.u_distribution = config.u_distribution
        self.u_dim = config.u_dim
        self.gen_cov()


    def gen_cov(self):
        r = self.config.gaussian_r
        m = self.config.x_dim
        self.sigmas = 0.5 * np.exp(-r * np.arange(m))
        self.cov = np.diag(self.sigmas ** 2)
    #     consider random eigenvectors matrix.

    def sample_x(self, eval=0):
        return torch.from_numpy(np.random.multivariate_normal(mean=np.zeros_like(self.sigmas), cov=self.cov, size=self.batch_size))

    def sample_u(self):
        if self.u_distribution == "gaussian":
            return self.config.u_std*torch.randn(size=(self.batch_size, self.u_dim))
        else:
            return torch.rand(size=(self.batch_size, self.u_dim))


class DataGenRDPGauss(object):
    def __init__(self, config):
        self.config = config
        self.batch_size = config.batch_size

    def sample_x(self, eval=0):
        if eval:
            return torch.randn(size=(5*self.batch_size, 1))
        else:
            return torch.randn(size=(self.batch_size, 1))

    def sample_u(self, eval=0):
        if eval:
            return self.config.u_std*torch.randn(size=(5*self.batch_size, 1))
        else:
            return self.config.u_std * torch.randn(size=(self.batch_size, 1))


class DataGenRDCategorical(object):
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.alphabet_size = config.alphabet_size
        self.bern_p = config.bern_p


    def sample_x(self, eval=None):
        if self.alphabet_size == 2:
            # sample bernoulli
            x = bernoulli_sample(self.batch_size, self.bern_p)
            return F.one_hot(x.squeeze().long(), num_classes=2).to(torch.float32)

        else:
            x = torch.from_numpy(np.random.choice(self.alphabet_size, self.batch_size))
            return torch.Tensor.float(nn.functional.one_hot(x, self.alphabet_size)).to(torch.float32)

        # return self.sampling_fn(self.batch_size, self.alphabet_size)
        # return torch.from_numpy(np.random.multinomial(n=self.batch_size, pvals=[1/self.alphabet_size]*self.alphabet_size))

        # x = torch.from_numpy(np.random.choice(self.alphabet_size, self.batch_size))
        # return torch.Tensor.float(nn.functional.one_hot(x, self.alphabet_size))


def bernoulli_sample(batch_size, p=0.5):
    return torch.bernoulli(p*torch.ones(size=(batch_size, 1)))


def categorical_uniform(batch_size, alphabet_size):
    x = torch.from_numpy(np.random.choice(alphabet_size, batch_size))
    return torch.Tensor.float(nn.functional.one_hot(x, alphabet_size))


class DataGenAWGN(object):
    def __init__(self, config):
        self.config = config

    def sample(self):
        x, z = torch.randn(size=(self.config.batch_size, self.config.x_dim)),\
               torch.randn(size=(self.config.batch_size, self.config.x_dim))
        y = x + z
        return x, y


# class DataGenMNIST(object):
#     def __init__(self, config):
#         self.config = config
#         self.batch_size = config.image_batch_size
#         self.u_distribution = config.u_distribution
#         self.u_dim = config.u_dim
#
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])
#         train_dataset = datasets.MNIST(root='./datasets/mnist', train=True, download=True, transform=transform)
#         # test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=self.transform)
#         self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
#         # self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
#
#     def sample_u(self, x):
#         return torch.rand(size=x.size())




