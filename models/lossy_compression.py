import torch
import torch.nn as nn
from torch.nn import functional as F
import huffman
from utilities import compute_huffman_entropy
from collections import Counter
import heapq
import numpy as np


# Implementation #1
class VAE_(nn.Module):
    def __init__(self, config, h_dim1=512, h_dim2=256, z_dim=2):
        super(VAE, self).__init__()

        # encoder part
        self.fc1 = nn.Linear(config.x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, config.x_dim)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)  # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h))

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        # return self.decoder(z), mu, log_var
        return self.decoder(z)


# Implementation #2 using both AE and VAE
class AE(nn.Module):
    def __init__(self, latent_dim):
        super(AE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, latent_dim)

        # Decoder
        self.fc4 = nn.Linear(latent_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc3(h2)

    def decode(self, z):
        h4 = F.relu(self.fc4(z))
        h5 = F.relu(self.fc5(h4))
        return torch.sigmoid(self.fc6(h5))

    def forward(self, x):
        z = self.encode(x.view(-1, 784))
        return self.decode(z)


class VAE(AE):
    def __init__(self, config):
        super(VAE, self).__init__(config.latent_dim)
        self.fc3_mean = nn.Linear(256, config.latent_dim)
        self.fc3_log_var = nn.Linear(256, config.latent_dim)
        self.out_vae_latent = (config.kl_loss == 'club')
        if config.quantize_alphabet > 0:
            raise Exception("Terminated. Irrelevant run for sweep, Q>0.")

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc3_mean(h2), self.fc3_log_var(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        y = self.decode(z).reshape((-1,1,28,28))
        if self.out_vae_latent:
            return y, mu, logvar, z
        else:
            return y



# Implementation of Blau & Michaeli RDP models

class BM_VAE(nn.Module):
    def __init__(self, config):
        super(BM_VAE, self).__init__()
        self.config = config
        self.encoder = BMEncoder(latent_dim=config.latent_dim)
        if config.quantize_alphabet:
            self.quantizer = Quantizer(num_centers=config.quantize_alphabet, num_features=config.latent_dim)
            val = 2 / (config.quantize_alphabet - 1)
            self.uniform = torch.distributions.uniform.Uniform(-val / 2, val / 2)
        self.decoder = BMDecoder(dim=config.latent_dim)



    def forward(self, x):
        z = self.encoder(x)


        if self.config.quantize_alphabet:
            q, _ = self.quantizer(z)
            if self.config.noise_encoder:
                q = q + self.uniform.sample(sample_shape=q.shape).cuda()
            y = self.decoder(q)
            return y
        else:
            if self.config.noise_encoder and self.config.quantize_alphabet:
                z = z + self.uniform.sample(sample_shape=z.shape).cuda()
            y = self.decoder(z)
            return y


    def entropy_coded(self, x):
        # obtain hard symbols from quantizer:
        z = self.encoder(x)
        _, symbols_hard = self.quantizer(z)

        # compute frequencies:
        symbols_hard_list = symbols_hard.tolist()
        flattened_symbols_hard_list = [item for sublist in symbols_hard_list for item in sublist]
        frequency = Counter(flattened_symbols_hard_list)

        heap = [[weight, [symbol, ""]] for symbol, weight in frequency.items()]
        heapq.heapify(heap)

        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

        huffman_codes = {symbol: code for symbol, code in
                         sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))}
        encoded_data = ''.join(huffman_codes[symbol] for symbol in flattened_symbols_hard_list)
        encoded_tensor = torch.tensor([int(bit) for bit in encoded_data])
        entropy = compute_huffman_entropy(encoded_tensor)
        upper_bound = self.config.latent_dim*np.log2(self.config.quantize_alphabet)
        return entropy, upper_bound





class BMEncoder(nn.Module):
    def __init__(self, latent_dim):
        """
        Implementation of the encoder network from Blau & Michaeli RDP paper.
        """
        super(BMEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.FF = self.build_ff()


    def forward(self, x):
        #  model assumed x is flattened
        x = x.reshape(-1, 784)
        x = self.FF(x)
        return x

    def build_ff(self):
        dims = [512, 256, 128, 128]
        seq = []
        dim_ = 784
        for i, dim in enumerate(dims):
            seq += [nn.Linear(dim_, dim), nn.BatchNorm1d(dim), nn.LeakyReLU()]
            dim_ = dim
        seq += [nn.Linear(dim_, self.latent_dim), nn.BatchNorm1d(self.latent_dim), nn.Tanh()]

        return nn.Sequential(*seq)




class Quantizer(nn.Module):
    def __init__(self, num_centers, num_features):
        super(Quantizer, self).__init__()
        self.num_centers = num_centers
        self.num_features = num_features
        self.sigma = 2./num_centers

        # Initialize "num_centers" centers to be evenly spaced in [-1, 1]^(num_features)
        # self.centers = nn.Parameter(torch.linspace(-1, 1, num_centers))
        self.centers = torch.Tensor(torch.linspace(-1, 1, num_centers)).cuda()

    def forward(self, x):
        # Compute distance between x and centers
        # x = x.unsqueeze(-1)
        # centers = self.centers.unsqueeze(0)
        dist = torch.square(x.unsqueeze(1) - self.centers.unsqueeze(0).unsqueeze(2))

        # Compute phi_soft and phi_hard
        phi_soft = torch.softmax(-self.sigma * dist, dim=1)
        phi_hard = torch.softmax(-1e7 * dist, dim=1)

        # Compute symbols_hard
        symbols_hard = torch.argmax(phi_hard, dim=1)

        # Compute softout and hardout
        softout = torch.sum(phi_soft * self.centers.unsqueeze(0).unsqueeze(2), dim=1)
        hardout = torch.sum(phi_hard * self.centers.unsqueeze(0).unsqueeze(2), dim=1)

        q = softout + hardout.detach() - softout.detach()

        return q, symbols_hard
        #
        #
        #
        #
        #
        #
        # # Compute phi_soft
        # phi_soft = F.softmax(-self.sigma * dist, dim=-1)
        #
        # # Compute soft quantized output
        # softout = torch.sum(phi_soft.unsqueeze(2) * self.centers, dim=1)
        #
        # # Compute phi_hard and symbols_hard in a no_grad block
        # with torch.no_grad():
        #     phi_hard = torch.argmax(F.softmax(-1e7 * dist, dim=-1), dim=-1)
        #     symbols_hard = F.one_hot(phi_hard, self.num_centers)
        #
        # # Compute hard quantized output
        # hardout = torch.sum(symbols_hard.unsqueeze(2) * self.centers, dim=1)
        #
        # # Use the hard quantized output for forward pass and
        # # the soft quantized output for backward pass
        # q = hardout + softout.detach() - softout.detach()
        #
        # reg_term = torch.sum(self.centers ** 2)  # not sure I want
        #
        # return q, symbols_hard


class BMDecoder(nn.Module):
    def __init__(self, dim):
        super(BMDecoder, self).__init__()
        self.dim = dim
        self.ffs = nn.Sequential(*[nn.Linear(dim, 128), nn.BatchNorm1d(128), nn.LeakyReLU(),
                                   nn.Linear(128, 512), nn.BatchNorm1d(512), nn.LeakyReLU()])
        self.unflatten = nn.Unflatten(1, (32, 4, 4))
        self.convs = nn.Sequential(*[nn.ConvTranspose2d(32, 64, kernel_size=5, stride=2, padding=0), nn.BatchNorm2d(64), nn.LeakyReLU(),
                                     nn.ConvTranspose2d(64, 128, kernel_size=5, stride=2, padding=0), nn.BatchNorm2d(128), nn.LeakyReLU(),
                                     nn.ConvTranspose2d(128, 1, kernel_size=4, stride=1, padding=0), nn.Sigmoid(),
                                     ])

    def forward(self, x):
        x = self.ffs(x)
        x = self.unflatten(x)
        x = self.convs(x)
        return x


class BMDiscriminator(nn.Module):
    def __init__(self):
        super(BMDiscriminator, self).__init__()
        self.convs = nn.Sequential(*[nn.Conv2d(1, 64, kernel_size=2, stride=2, padding=0), nn.ReLU(),
                      nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=0), nn.ReLU(),
                      nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0), nn.ReLU()])
        self.linear = nn.Linear(4096,1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, 4096)
        x = self.linear(x)
        return x