import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import numpy as np
from torch.utils.data import Dataset


root = 'data/'


def GetDataLoader(config, for_eval=None):
    if config.data == 'MNIST':
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor()])

        dataset = dsets.MNIST(root+'mnist/',
                                download=False, transform=transform)

    elif config.data == 'GaussianLoader':
        dataset = GaussianDataset(config)


    if for_eval:
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=60000,
                                                 shuffle=True)
    else:
        dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=config['batch_size'],
                                             shuffle=True)

    return dataloader


class GaussianDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.num_samples = config.num_samples
        self.data_gen = GaussianGenerator(config)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data_gen.sample_x(), 0


def PreprocessBatch(data):
    # compute batch statistics
    mean = data.mean()
    std = data.std()

    # normalize batch
    data = (data - mean) / std

    return data



class GaussianGenerator(object):
    def __init__(self, config):
        self.config = config
        self.batch_size = config.batch_size
        self.gen_cov()

    def gen_cov(self):
        r = self.config.gaussian_r
        m = self.config.x_dim
        self.sigmas = 0.5 * np.exp(-r * np.arange(m))
        self.cov = np.diag(self.sigmas ** 2)

    def sample_x(self):
        return torch.from_numpy(np.random.multivariate_normal(mean=np.zeros_like(self.sigmas), cov=self.cov))


