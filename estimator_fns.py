import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

"""
Some of the estimators functions are taken from "Understanding the Limitations of Variational Mutual Information Estimators
" By Jiaming Song, Stefano Ermon.
"""

def logmeanexp(x, device='cuda'):
    """Compute logmeanexp of x."""
    batch_size = x.size(0)
    logsumexp = torch.logsumexp(x, dim=(0, 1))
    num_elem = batch_size

    return logsumexp - torch.log(torch.tensor(num_elem).float()).to(device)


def logmeanexp_diag(x, device='cuda'):
    """Compute logmeanexp over the diagonal elements of x."""
    batch_size = x.size(0)

    logsumexp = torch.logsumexp(x.diag(), dim=(0,))
    num_elem = batch_size

    return logsumexp - torch.log(torch.tensor(num_elem).float()).to(device)


def logmeanexp_nodiag(x, dim=None, device='cuda'):
    batch_size = x.size(0)
    if dim is None:
        dim = (0, 1)

    logsumexp = torch.logsumexp(
        x - torch.diag(np.inf * torch.ones(batch_size).to(device)), dim=dim)

    try:
        if len(dim) == 1:
            num_elem = batch_size - 1.
        else:
            num_elem = batch_size * (batch_size - 1.)
    except ValueError:
        num_elem = batch_size - 1
    return logsumexp - torch.log(torch.tensor(num_elem)).to(device)


def tuba_lower_bound(scores, log_baseline=None):
    if log_baseline is not None:
        scores -= log_baseline[:, None]

    # First term is an expectation over samples from the joint,
    # which are the diagonal elmements of the scores matrix.
    joint_term = scores.diag().mean()

    # Second term is an expectation over samples from the marginal,
    # which are the off-diagonal elements of the scores matrix.
    marg_term = logmeanexp_nodiag(scores).exp()
    return 1. + joint_term - marg_term


def nwj_lower_bound(scores):
    return tuba_lower_bound(scores - 1.)


def infonce_lower_bound(scores):
    nll = scores.diag().mean() - scores.logsumexp(dim=1)
    # Alternative implementation:
    # nll = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=tf.range(batch_size))
    mi = torch.tensor(scores.size(0)).float().log() + nll
    mi = mi.mean()
    return mi


def js_fgan_lower_bound(f):
    """Lower bound on Jensen-Shannon divergence from Nowozin et al. (2016)."""
    f_diag = f.diag()
    first_term = -F.softplus(-f_diag).mean()
    n = f.size(0)
    second_term = (torch.sum(F.softplus(f)) -
                   torch.sum(F.softplus(f_diag))) / (n * (n - 1.))
    return first_term - second_term


def js_lower_bound(f):
    """Obtain density ratio from JS lower bound then output MI estimate from NWJ bound."""
    nwj = nwj_lower_bound(f)
    js = js_fgan_lower_bound(f)

    with torch.no_grad():
        nwj_js = nwj - js

    return js + nwj_js


def dv_upper_lower_bound(f):
    """
    Donsker-Varadhan lower bound, but upper bounded by using log outside. 
    Similar to MINE, but did not involve the term for moving averages.
    """
    first_term = f.diag().mean()
    second_term = logmeanexp_nodiag(f)

    return first_term - second_term


def mine_lower_bound(f, buffer=None, momentum=0.9):
    """
    MINE lower bound based on DV inequality. 
    """
    if buffer is None:
        buffer = torch.tensor(1.0).cuda()
    first_term = f.diag().mean()

    buffer_update = logmeanexp_nodiag(f).exp()
    with torch.no_grad():
        second_term = logmeanexp_nodiag(f)
        buffer_new = buffer * momentum + buffer_update * (1 - momentum)
        buffer_new = torch.clamp(buffer_new, min=1e-4)
        third_term_no_grad = buffer_update / buffer_new

    third_term_grad = buffer_update / buffer_new

    return first_term - second_term - third_term_grad + third_term_no_grad, buffer_update


def smile_lower_bound(f, clip=None):
    if clip is not None:
        f_ = torch.clamp(f, -clip, clip)
    else:
        f_ = f
    z = logmeanexp_nodiag(f_, dim=(0, 1))
    dv = f.diag().mean() - z


    #########################
    # original implementation:
    ##########################
    js = js_fgan_lower_bound(f)

    with torch.no_grad():
        dv_js = dv - js

    return js + dv_js

    #########################
    # paper implementation
    #########################
    # return dv

########################################
# My additions:
def dv_ref(f, f_ref):
    """
    DV lower bound, adapted to have different of f and f_ samples number, which are not given in a 2d structure.
    """
    first_term = f.mean()
    second_term = logmeanexp(f_ref)

    return first_term - second_term


def smile_loss(f, clip=None):
    if clip is not None:
        f_ = torch.clamp(f, -clip, clip)
    else:
        f_ = f
    z = logmeanexp_nodiag(f_, dim=(0, 1))
    dv = f.diag().mean() - z

    return dv


def regularized_dv(f, factor=None):
    """
    Donsker-Varadhan lower bound, but upper bounded by using log outside.
    Similar to MINE, but did not involve the term for moving averages.
    """
    first_term = f.diag().mean()
    second_term = logmeanexp_nodiag(f)

    mi = first_term - second_term

    reg = torch.square(second_term)

    with torch.no_grad():
        reg_ = reg

    return mi - reg + reg_  # should present DV as the outcome but take derivative wrt regularized DV


def log_loss(f):
    # log loss for club-based scheme (reparametrization trick)
    mu, logvar, y_samples = f  # unpack f
    mi = (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    return mi


def club_loss(f):
    # calculate club loss based on the unnormalized log likelihood estaimted model
    mu, logvar, y_samples = f  # unpack f
    positive = - (mu - y_samples)**2 /2./logvar.exp()
    prediction_1 = mu.unsqueeze(1)  # shape [nsample,1,dim]
    y_samples_1 = y_samples.unsqueeze(0)  # shape [1,nsample,dim]

    # log of conditional probability of negative sample pairs
    negative = - ((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2. / logvar.exp()

    return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()


def FLO(scores):
    ########
    # K-FLO implement:
    u = scores['u']
    g = scores['g']
    g0 = scores['g0']
    exponent = (torch.exp(-u + g0.t() - g)).mean()
    loss = u + exponent -1
    #########
    # 1-FLO implement:
    # loss = scores['u'] + torch.exp(-scores['u'] + scores['g0'] - scores['g']) - 1
    return -loss.mean()


def alt_discrete_loss(scores, p, y):
    """
    calculate alternative optimization loss for discrete optimization case.
    """
    t = scores.diag()

    # b = p.shape[0]
    # n_ind = torch.arange(0, b).unsqueeze(-1).long()
    # y_ind = torch.cat([n_ind, y], dim=-1)

    lp = torch.log(p.gather(1, y).squeeze())
    loss = (lp * (t - t.mean())).mean()
    return loss


def estimate_mutual_information(estimator, x, y, critic_fn, club_flag=None,
                                baseline_fn=None, y_int = None, **kwargs):
    """Estimate variational lower bounds on mutual information.

  Args:
    estimator: string specifying estimator, one of:
      'nwj', 'infonce', 'tuba', 'js', 'interpolated'
    x: [batch_size, dim_x] Tensor
    y: [batch_size, dim_y] Tensor
    critic_fn: callable that takes x and y as input and outputs critic scores
      output shape is a [batch_size, batch_size] matrix
    baseline_fn (optional): callable that takes y as input 
      outputs a [batch_size]  or [batch_size, 1] vector
    alpha_logit (optional): logit(alpha) for interpolated bound

  Returns:
    scalar estimate of mutual information
    """

    # ref samples
    if estimator == 'dy':
        y_ref = sample_ref(y)
        scores_y = critic_fn(y)
        scores_y_ref = critic_fn(y_ref)
        return dv_ref(scores_y, scores_y_ref)
    elif estimator == 'dxy':
        dup = 10  #TD - take care of dup
        y_ref = sample_ref(y, dup=dup)
        scores_xy = critic_fn(x, y)
        x = torch.cat([x] * dup, dim=0)
        scores_xy_ref = critic_fn(x, y_ref)
        return dv_ref(scores_xy, scores_xy_ref)
    elif estimator == 'FLO':
        y0 = y[torch.randperm(y.size()[0])]
        scores = {}
        scores['g'] = critic_fn['g'](x, y)
        scores['g0'] = critic_fn['g'](x, y0)
        scores['u'] = critic_fn['u'](x, y)
        return FLO(scores)




    # original kl cases
    scores = critic_fn(x, y)
    # los calc:
    if baseline_fn is not None:
        # Some baselines' output is (batch_size, 1) which we remove here.
        log_baseline = torch.squeeze(baseline_fn(y))
    if estimator == 'infonce':
        mi = infonce_lower_bound(scores)
    elif estimator == 'nwj':
        mi = nwj_lower_bound(scores)
    elif estimator == 'tuba':
        mi = tuba_lower_bound(scores, log_baseline)
    elif estimator == 'js':
        mi = js_lower_bound(scores)
    elif estimator == 'smile':
        mi = smile_lower_bound(scores, **kwargs)
    elif estimator == 'dv':
        mi = dv_upper_lower_bound(scores)
    elif estimator == 'smile_loss':
        mi = smile_loss(scores)
    elif estimator == 'regularized_dv':
        mi = regularized_dv(scores)
    elif estimator == 'CLUB':
        scores = (scores[0], scores[1], y)
        if club_flag == 'q_loss':
            mi = log_loss(scores)
        else:
            mi = club_loss(scores)
    elif estimator == 'alt_loss':
        mi = alt_discrete_loss(scores, x, y_int)
    return mi


def sample_ref(y, dup=10):
    # generate bounding box
    batch_size = y.shape[0]
    dim = y.shape[1]
    min_vals, _ = torch.min(y, dim=0)
    max_vals, _ = torch.max(y, dim=0)
    bounding_box = torch.stack([min_vals, max_vals])

    # sample bounding box:
    uniform_samples = torch.rand(batch_size*dup, dim).cuda()
    y_ref = bounding_box[0] + uniform_samples * (bounding_box[1] - bounding_box[0])

    return y_ref