import numpy as np
from scipy.optimize import bisect
from config import GetConfig
from args_utils import GetArgs
import torch
import wandb
import math as m


# Preprocessing Utils:
def PreprocessMeta():
    """
    steps:
    0. get config
    1. parse args
    2. initiate wandb
    """
    args = GetArgs()
    config = GetConfig(args)
    # add wandb:
    wandb_proj = "RD" if not hasattr(config, 'wandb_project_name') else config.wandb_project_name
    if config.using_wandb:
        wandb.init(project=wandb_proj,
                   entity="dortsur",
                   config=config)
    return config





# Gaussian Utils:
def GaussianRateDistortion(D, sigmas):
    # sigmas are the singular values of the Gaussian random vector covariance matrix
    optimal_gamma = ReverseWaterFilling(sigmas, D)
    ind_over = sigmas ** 2 > optimal_gamma
    return np.sum(0.5 * np.log(sigmas[ind_over] ** 2 / optimal_gamma))


def gamma_fn(lam, sigmas, D):
    ind_under = sigmas ** 2 <= lam
    lhs = lam * np.sum(1 - ind_under) + np.sum(sigmas[ind_under] ** 2)
    return lhs - D

def ReverseWaterFilling(sigmas, D):
    # reverse waterfilling operation, return lambda
    if D > sum(sigmas ** 2):
        return max(sigmas ** 2)
    objective = lambda gamma: gamma_fn(gamma, sigmas, D)
    lam_opt = bisect(objective, 0, D)
    return lam_opt

def Wasserstein1D(x,y, p=2):
    x_sorted, _ = torch.sort(x, dim=0)
    y_sorted, _ = torch.sort(y, dim=0)
    return (x_sorted-y_sorted).pow(p).mean()

def Wasserstein_KR(x,y,h):
    """
    calculate the loss for Wasserstein regularization through Krnatarovich-Rubenstein duality
    """
    return h(x).mean()-h(y).mean()

def GaussianRateDistortionPerception(D, P):
    return rdp_(D=D, P=P, s=1)
    # if np.sqrt(P) < 1 - np.sqrt(np.abs(1-D)):
    #     numerator = (1-np.sqrt(P))**2
    #     denominator = (1-np.sqrt(P))**2 - 0.5*(1+(1-np.sqrt(P))**2-D)**2
    #     return 0.5*np.log(numerator/denominator)
    # else:
    #     return np.max(0.5*np.log(1/D), 0)


def rdp_(D,P,s):
  if P**0.5 < s - m.sqrt(abs(s**2-D)):
    a = (s - P**0.5 ) **2
    numerator = s**2 * a
    denominator = s**2 * a - ((s**2 + a - D ) /2)**2
    return 0.5*m.log(numerator/denominator)
  else:
    return max(0, 0.5*m.log(s**2/D))
#####################################

# def sample_u(batch_size, u_dim, u_distribution='gaussian'):
#     if u_distribution == "gaussian":
#         return torch.randn(size=(batch_size, u_dim))
#     else:
#         return torch.rand(size=(batch_size, u_dim))
#
#
# def sample_x(batch_size, x_dim, base_distribution, cov, sigmas):
#     if base_distribution == "gaussian_sigma_decay":
#         return torch.from_numpy(np.random.multivariate_normal(mean=np.zeros_like(sigmas), cov=cov, size=batch_size))
#     else:
#         return torch.randn(size=(batch_size, x_dim))
#
#
# def sample_awgn(batch_size, dim):
#     x, z = torch.randn(size=(batch_size, dim)), torch.randn(size=(batch_size, dim))
#     y = x + z
#     return x, y
#
#
# def calcGaussianDecay(D, x_dim):
#     r = 0.3
#     m = x_dim
#     sigmas = 0.5 * np.exp(-r * np.arange(m))
#     cov = np.diag(sigmas ** 2)
#     expected_sol = GaussianRateDistortion(sigmas=sigmas, D=D)
#
#     return cov, sigmas, expected_sol


def hamming_rd(alphabet, D, p=None):
    # return m.log(alphabet) - bin_ent(D) - D * m.log(alphabet - 1)
    if alphabet == 2:
        if D < p:
            return bin_ent(p) - bin_ent(D)
        else:
            return 0
    else:
        return m.log(alphabet) - bin_ent(D) - D*m.log(alphabet-1)



def bin_ent(p):
    return -p*m.log(p)-(1-p)*m.log(1-p)


def tri_ent(p, q):
    return -p*m.log(p)-q*m.log(q)-(1-p-q)*m.log(1-p-q)



def hamming_rdp(D, P, p=0.5):
    if p < P:
        raise ValueError("Bernoulli parameter p Should be BIGGER than perception constraint P")
    q= 1-p
    D1 = P/(1-2*(p-P))
    D2 = 2*p*q - (q-p)*P
    if D < D1:
        return bin_ent(p) - bin_ent(D)
    elif D < D2:
        return 2*bin_ent(p)+bin_ent(p-P)-tri_ent((D-P)/2, p)-tri_ent((D+P)/2, q)
    else:
        return 0.0


def compute_huffman_entropy(symbols_hard):
    # Compute the frequency of each symbol
    unique, counts = symbols_hard.unique(return_counts=True)

    # Compute the probabilities
    probabilities = counts.float() / symbols_hard.numel()

    # Compute the entropy
    entropy = -torch.sum(probabilities * torch.log(probabilities))

    return entropy



