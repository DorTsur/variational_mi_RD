import torch.nn as nn
import torch
import torch.nn.functional as F


######### Discrete output model for alternative loss ##########
class PMFModel(nn.Module):
    def __init__(self, config):
        super(PMFModel, self).__init__()
        # self.hidden_dims = [32, 32, 32, 32]
        d = max(2*config.alphabet_size, 32)
        self.hidden_dims = [d, d, 2*d, d, d]
        self.config = config

        self.ff_pass = [nn.Linear(self.config.alphabet_size, self.hidden_dims[0]), nn.ELU()]
        d_ = self.hidden_dims[0]
        for d in self.hidden_dims[1:]:
            self.ff_pass += [nn.Linear(d_, d), nn.ELU()]
            d_ = d
        self.ff_pass += [nn.Linear(d_, self.config.alphabet_size)]
        self.ff_pass = nn.Sequential(*self.ff_pass)

        self.sm = nn.Softmax(dim=1)


    def forward(self, x):
        # input is one hot
        # x_oh = F.one_hot(x.to(torch.int64), num_classes=self.config.alphabet_size).float()
        # feedforwards , input and output dims are alphabet size
        p = self.ff_pass(x)
        # softmax
        p = self.sm(p)
        return p.to(torch.float32)


class SamplerModel(nn.Module):
    def __init__(self, config):
        super(SamplerModel, self).__init__()
        self.alphabet_size = config.alphabet_size

    def forward(self, p):
        return torch.multinomial(p, num_samples=1)



######### Gumbel Softmax Implementation ##########
## IN PROGRESS
class DiscreteNDT(nn.Module):
    def __init__(self, config):
        super(DiscreteNDT, self).__init__()
        self.logits_model = nn.Sequential(
            nn.Linear(config.x_dim, config.ndt_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.ndt_hidden_dim, config.ndt_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.ndt_hidden_dim, config.y_dim)
        )
        self.gumbel_softmax = GumbelSoftmax(config)
        self.ste_onehot = OneHotSTE()

    def forward(self, x):
        l = self.logits_model(x)
        y = self.gumbel_softmax(l)
        y = self.ste_onehot(y)
        return l, y


class GumbelSoftmax(nn.Module):
    def __init__(self, config):
        super(GumbelSoftmax, self).__init__()
        #Gumbel softmax implementation
        self.T = config.gsm_temperature

    def forward(self, l):
        """
        1. sample gumbel
        2. crate softmax inputs
        3. apply softmax
        """
        g = self.sample_gumbel(l.shape)
        h = (g + l) / self.T
        p = nn.functional.softmax(h)
        return p

    def sample_gumbel(self, shape, eps=1e-20):
        unif = torch.rand(*shape).cuda()
        g = -torch.log(-torch.log(unif + eps))
        return g


class OneHotSTEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Assuming input is in the shape (batch_size, sequence_length),
        # and each element in the sequence is an integer index.
        return torch.nn.functional.one_hot(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class OneHotSTE(torch.nn.Module):
    def forward(self, input):
        return OneHotSTEFunction.apply(input)