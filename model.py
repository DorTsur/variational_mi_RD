import torch
import torch.nn as nn
from typing import Optional, Dict
import torch.nn.functional as F
from models.discrete_opt_models import DiscreteNDT, PMFModel, SamplerModel
from models.lossy_compression import BM_VAE, VAE, VAE_



def GetModel(config):
    if config.model == 'MINE':
        if config.kl_loss == 'FLO':
            model = {
                'kl': {
                    'g': FLO_Wrapper(FLO_MLP(config.x_dim+config.y_dim)).cuda(),
                    'u': FLO_Wrapper(FLO_MLP(config.x_dim+config.y_dim)).cuda()
                }
            }
        else:
            model = {
            'kl': SeparableCritic(config).cuda()
            }
    elif config.model == 'MINE_NDT':
        model = {
            'kl': SeparableCritic(config).cuda(),
            'ndt': GaussianNDT(config.x_dim, config.u_dim, config.y_dim, config.D).cuda()
        }
        if config.kl_loss == 'FLO':
            model['kl'] = {
                    'g': FLO_Wrapper(FLO_MLP(config.x_dim+config.y_dim)).cuda(),
                    'u': FLO_Wrapper(FLO_MLP(config.x_dim+config.y_dim)).cuda()
                }
        elif config.kl_loss == 'CLUB':
            model['kl'] = CLUBCritic(config).cuda()
        else:
            model['kl'] = SeparableCritic(config).cuda()
    # elif config.model == 'MINE_NDT_discrete':
    #     model = {
    #         'kl': SeparableCritic(config).cuda(),
    #         'ndt': DiscreteNDT(config).cuda()
    #     }
    elif config.model == 'MNIST':
        model = {
            'kl': ConvSeparableCritic().cuda(),
            'ndt': ConvAutoencoder(config).cuda()
        #     TD - MODELS
        }
    elif config.model == 'MNIST_VAE':
        model = {
            'kl': SeparableCritic(config).cuda(),
            'ndt': VAE(config).cuda()
        #     TD - MODELS
        }
    elif config.model == 'MNIST_VAE_BM':
        model = {
            'kl': SeparableCritic(config).cuda(),
            'ndt': BM_VAE(config).cuda()
        #     TD - MODELS
        }
    elif config.model == 'DiscreteAlt':
        model = {
            'kl': SeparableCritic(config).cuda(),
            'ndt': PMFModel(config).cuda(),
            'sampler': SamplerModel(config).cuda()
        }
    else:
        raise ValueError("'{}' is an invalid model name")
    return model


######### Continuous vectors MINE and NDT models ##########
def mlp(dim, hidden_dim, output_dim, layers, activation):
    """Create a mlp from the configurations."""
    activation = {
        'relu': nn.ReLU,
        'elu': nn.ELU,
        'gelu': nn.GELU,
        'tanh': nn.Tanh,
        'sigmoid':  nn.Sigmoid
    }[activation]

    seq = [nn.Linear(dim, hidden_dim), activation()]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), activation()]
    seq += [nn.Linear(hidden_dim, output_dim)]

    return nn.Sequential(*seq)


# Vector FF NDT model:
class GaussianNDT(nn.Module):
    def __init__(self, x_dim, u_dim, y_dim, D, constrained=True):
        super(GaussianNDT, self).__init__()
        self.constrained = constrained
        hidden_dim = 2 * max(16,x_dim)
        # prev net hidden dim:
        # self.xu_to_features = nn.Sequential(
        #     nn.Linear(x_dim+u_dim, y_dim),
        #     nn.ReLU(),
        #     nn.Linear(y_dim, y_dim),
        #     nn.ReLU(),
        #     nn.Linear(y_dim, y_dim)
        # )
        self.xu_to_features = nn.Sequential(
            nn.Linear(x_dim + u_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, y_dim)
        )
        print(f'bigger net')

        self.constraint_layer = DistortionMSELayer(D)


    def forward(self, x, u):
        input_data = torch.cat((x, u), -1)
        y = self.xu_to_features(input_data)
        if self.constrained:
            y = self.constraint_layer(x, y)
        return y
        # input_data = torch.cat((x, u), -1)
        # z = self.xu_to_features(input_data)
        # y = self.constraint_layer(x, z)
        # return y


# Distortion constraint layer under MSE distortion
class DistortionMSELayer(torch.nn.Module):
    def __init__(self, D, dim=None):
        super(DistortionMSELayer, self).__init__()
        self.D = D

    def forward(self, x, y):
        distortion = torch.mean(torch.square(torch.linalg.norm(x - y, dim=-1, keepdim=True)))
        return x - torch.sqrt(self.D / distortion) * (x - y)


# Standard separable critic for MI estimation
class SeparableCritic(nn.Module):
    """Separable critic. where the output value is g(x) h(y). """

    def __init__(self, config):
        super(SeparableCritic, self).__init__()
        if config.data == 'Discrete':
            dim = config.alphabet_size
        else:
            dim = config.x_dim
        self._g = mlp(dim, config.critic_hidden_dim, config.critic_embed_dim, config.critic_layers, config.critic_activation)
        self._h = mlp(dim, config.critic_hidden_dim, config.critic_embed_dim, config.critic_layers, config.critic_activation)

    def forward(self, x, y):
        scores = torch.matmul(self._h(y), self._g(x).t())
        return scores


# Standard concat critic for MI estimation
class ConcatCritic(nn.Module):
    """Concat critic, where we concat the inputs and use one MLP to output the value."""

    def __init__(self, dim, hidden_dim, layers, activation, **extra_kwargs):
        super(ConcatCritic, self).__init__()
        # output is scalar score
        self._f = mlp(dim * 2, hidden_dim, 1, layers, activation)

    def forward(self, x, y):
        batch_size = x.size(0)
        # Tile all possible combinations of x and y
        x_tiled = torch.stack([x] * batch_size, dim=0)
        y_tiled = torch.stack([y] * batch_size, dim=1)
        # xy is [batch_size * batch_size, x_dim + y_dim]
        xy_pairs = torch.reshape(torch.cat((x_tiled, y_tiled), dim=2), [
                                 batch_size * batch_size, -1])
        # Compute scores for each x_i, y_j pair.
        scores = self._f(xy_pairs)
        return torch.reshape(scores, [batch_size, batch_size]).t()



###########################################################
######### Additional models - modve to designated #########
###########################################################


class NoiseLayer(nn.Module):
    def __init__(self, config):
        super(NoiseLayer, self).__init__()
        self.std = config.u_std

    def forward(self, x):
        # add centered Gaussian noise to x.
        return x + (self.std * torch.randn(size=x.shape)).cuda()


class CLUBCritic(nn.Module):
    def __init__(self, config):
        super(CLUBCritic, self).__init__()
        self.config = config
        self.p_mu = nn.Sequential(nn.Linear(config.x_dim, config.critic_hidden_dim // 2),
                                  nn.ReLU(),
                                  nn.Linear(config.critic_hidden_dim // 2, config.y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(config.x_dim, config.critic_hidden_dim // 2),
                                      nn.ReLU(),
                                      nn.Linear(config.critic_hidden_dim // 2, config.y_dim),
                                      nn.Tanh())

    def forward(self, x, y=None):
        mu = self.p_mu(x)
        logvar = self.p_logvar(x)
        return mu, logvar






#################################################
#################################################
######### OLD MODELS - REMOVE OR DELETE #########
#################################################

class FLOCritic(nn.Module):
    """Separable critic. where the output value is g(x) h(y). """

    def __init__(self, dim, hidden_dim, embed_dim, layers, activation, **extra_kwargs):
        super(FLOCritic, self).__init__()
        self._g = mlp(dim, hidden_dim, embed_dim, layers, activation)
        self._h = mlp(dim, hidden_dim, embed_dim, layers, activation)

    def forward(self, x, y):
        g_scores = self.self._g(x, y)
        u_scores = self.self._h(x, y)
        y0 = y[torch.randperm(y.size()[0])]
        g0_scores = []
        for _ in self.K:
            g0_scores.append(self.self._g(x, y0))
        return g_scores, torch.cat(g0_scores, dim=-1), u_scores


class CriticY(nn.Module):
    """Separable critic. where the output value is g(x) h(y). """

    def __init__(self, dim, hidden_dim, embed_dim, layers, activation, **extra_kwargs):
        super(CriticY, self).__init__()
        self._g = mlp(dim, hidden_dim, embed_dim, layers, activation)

    def forward(self, x):
        return self._g(x)


class CriticXY(nn.Module):
    """Separable critic. where the output value is g(x) h(y). """

    def __init__(self, dim, hidden_dim, embed_dim, layers, activation, **extra_kwargs):
        super(CriticXY, self).__init__()
        self._g = mlp(dim, hidden_dim, embed_dim, layers, activation)
        self._h = mlp(dim, hidden_dim, embed_dim, layers, activation)

    def forward(self, x, y):
        scores = self._h(y)*self._g(x)
        return scores










class DistortionMSEImageLayer(torch.nn.Module):
    def __init__(self, D, dim=None):
        super(DistortionMSEImageLayer, self).__init__()
        self.D = D

    def forward(self, x, y):
        # Compute the squared difference between corresponding images in batches x and y
        squared_diff = (x - y) ** 2

        # Sum the squared differences across the channel, height, and width dimensions
        sum_squared_diff = torch.sum(squared_diff, dim=[1, 2, 3], keepdim=True)

        # Take the square root to compute the Euclidean distance (norm) between images
        norm_diff = torch.sqrt(sum_squared_diff)

        # Compute the mean distortion across the batch dimension
        distortion = torch.mean(norm_diff)

        return x - torch.sqrt(self.D / distortion) * (x - y)






######### FLO models ##############

class FLO_MLP(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=[512, 512], act_func=nn.ReLU()):
        super(FLO_MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.act_func = act_func

        layers = []
        for i in range(len(hidden_dim)):
            if i == 0:
                layer = nn.Linear(input_dim, hidden_dim[i])
            else:
                layer = nn.Linear(hidden_dim[i - 1], hidden_dim[i])
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            # layers.append(nn.ReLU(True))
            layers.append(act_func)
        if len(hidden_dim):  # if there is more than one hidden layer
            layer = nn.Linear(hidden_dim[-1], output_dim)
        else:
            layer = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
        layers.append(layer)

        self._main = nn.Sequential(*layers)

    def forward(self, x):
        out = x.view(x.shape[0], self.input_dim)
        out = self._main(out)
        return out


class FLO_Wrapper(nn.Module):
    def __init__(self, func):
        super(FLO_Wrapper, self).__init__()
        self.func = func

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)

        return self.func(xy)

######### FDV models ##############

# class BilinearFDVNCE(nn.Module):
#     def __init__(self,
#                  critic: nn.Module,
#                  args: Optional[Dict] = None,
#                  cuda: Optional[int] = None) -> None:
#
#         super(BilinearFDVNCE, self).__init__()
#         self.critic = critic
#         self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
#
#     def forward(self, x, y, y0=None, K=None):
#
#         '''
#         x:    n x p
#         y:    n x d true
#         y0:   n x d fake
#         '''
#         #         if K is None:
#         #             K = self.K
#         #         g = self.critic(x, y)
#         #         g0 = self.critic(x, y0)
#         #         u  = self.u_func(x, y)
#         #         output = u + torch.exp(-u+g0-g) - 1
#         output = self.PMI(x, y)
#         output = torch.clamp(output, -5, 15)
#         return output.mean()
#
#     def MI(self, x, y, K=10):
#         mi = 0
#         for k in range(K):
#             y0 = y[torch.randperm(y.size()[0])]
#             mi += self.forward(x, y, y0)
#
#         return -mi / K
#
#     def PMI(self, x, y, y0=None, K=None):
#         '''
#         x:    n x p
#         y:    n x d true
#         y0:   n x d fake
#         '''
#         gxy = self.critic(x, y)
#         if isinstance(gxy, tuple):
#             hx, hy = gxy
#             similarity_matrix = hx @ hy.t()
#
#             pos_mask = torch.eye(hx.size(0), dtype=torch.bool)
#             g = similarity_matrix[pos_mask].view(hx.size(0), -1)
#             g0 = similarity_matrix[~pos_mask].view(hx.size(0), -1)
#
#             logits = g0 - g
#
#             slogits = torch.logsumexp(logits, 1).view(-1, 1)
#
#             labels = torch.tensor(range(hx.size(0)), dtype=torch.int64).to(device)
#             dummy_ce = self.criterion(similarity_matrix, labels) - torch.log(torch.Tensor([hx.size(0)]).to(device))
#             dummy_ce = dummy_ce.view(-1, 1)
#
#             output = dummy_ce.detach() + torch.exp(slogits - slogits.detach()) - 1
#
#
#         #             g0_logsumexp = torch.logsumexp(g0,1).view(-1,1)
#
#         #             output = u + torch.exp(-u+g0_logsumexp-g)/(hx.size(0)-1) - 1
#
#         else:
#             output = None
#         #                 g = gxy
#         #                 if K is not None:
#
#         #                     for k in range(K-1):
#
#         #                         if k==0:
#         #                             y0 = y0
#         #                             g0,_ = torch.chunk(self.critic(x,y0),2,dim=1)
#         #                         else:
#         #                             y0 = y[torch.randperm(y.size()[0])]
#         #                             g00,_ = torch.chunk(self.critic(x,y0),2,dim=1)
#         #                             g0 = torch.cat((g0,g00),1)
#
#         #                     g0_logsumexp = torch.logsumexp(g0,1).view(-1,1)
#         #                     output = u + torch.exp(-u+g0_logsumexp-g)/(K-1) - 1
#         #                 else:
#
#         #                     g0, _ = torch.chunk(self.critic(x,y0),2,dim=1)
#         #                     output = u + torch.exp(-u+g0-g) - 1
#         return output



######## Images Nets #########

####### 1. MINE convolutional model
class ConvSeparableCritic(nn.Module):
    def __init__(self):
        super(ConvSeparableCritic, self).__init__()
        self._g = ConvNet()
        self._h = ConvNet()

    def forward(self, x, y):
        scores = torch.matmul(self._h(y), self._g(x).t())
        return scores

class ConvNet(nn.Module):
    def __init__(self, channels=64, kernel_size=5, input_dim=392, output_dim=256,
                 activation=torch.nn.functional.relu):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=kernel_size,
                               stride=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=2*channels, kernel_size=kernel_size,
                               stride=2, padding=2)
        self.norm = LayerNorm()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2*channels, output_dim)
        self.activation = activation

    def forward(self, x):
        x = x.view(-1, 1, 14, 28)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.norm(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class LayerNorm(nn.Module):
    """ Layer Normalization in the style of https://arxiv.org/abs/1607.06450 """
    def __init__(self, scale_initializer=None, bias_initializer=None,
                 axes=(1, 2, 3), epsilon=1e-6):
        super(LayerNorm, self).__init__()
        self.epsilon = epsilon
        self.axes = axes
        if scale_initializer is None:
            self.scale_initializer = nn.init.ones_
        if bias_initializer is None:
            self.bias_initializer = nn.init.zeros_

    def build(self, input_shape):
        self.scale = nn.Parameter(self.scale_initializer(torch.empty(input_shape[-1]))).to('cuda')
        self.bias = nn.Parameter(self.bias_initializer(torch.empty(input_shape[-1]))).to('cuda')

    def forward(self, x):
        if not hasattr(self, 'scale'):
            self.build(x.shape)
        mean = x.mean(self.axes, keepdim=True)
        std = x.std(self.axes, keepdim=True, unbiased=False)
        norm = (x - mean) * (1 / (std + self.epsilon))
        return norm * self.scale + self.bias


###### 2. NDT via Deep CAE
# CAE implementation from
# https://github.com/udacity/deep-learning-v2-pytorch/blob/master/autoencoder/convolutional-autoencoder/Convolutional_Autoencoder_Solution.ipynb
class ConvAutoencoder(nn.Module):
    def __init__(self, config):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 1 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

        self.constraint_layer = DistortionMSEImageLayer(config.D)
        self.add_noise = NoiseLayer(config)

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        z = F.relu(self.conv1(x))
        z = self.pool(z)
        # add second hidden layer
        z = F.relu(self.conv2(z))
        # z = F.relu(self.conv2(self.add_noise(z)))  # adding noise in the middle.
        z = self.pool(z)  # compressed representation

        # add the noise in the latent space


        ## decode ##
        # add transpose conv layers, with relu activation function
        z = F.relu(self.t_conv1(z))
        # output layer (with sigmoid for scaling from 0 to 1)
        z = F.sigmoid(self.t_conv2(z))

        ###
        # z = self.constraint_layer(x, z)
        # return y
        return z



