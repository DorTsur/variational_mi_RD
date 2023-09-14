import argparse


def GetArgs():
    """" collects command line arguments """

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--config', default=None, type=str, help='configuration file')
    argparser.add_argument('--experiment', default=None, type=str, help='experiment name')
    argparser.add_argument('--model', default=None, type=str, help='model name')
    argparser.add_argument('--trainer', default=None, type=str, help='trainer name')
    argparser.add_argument('--data', default=None, type=str, help='data name')

    argparser.add_argument('--x_dim', default=None, type=int, help='x dimension')
    argparser.add_argument('--y_dim', default=None, type=int, help='y dimension')

    argparser.add_argument('--u_dim', default=None, type=int, help='u dimension')
    argparser.add_argument('--u_distribution', default=None, type=str, help='u distribution')
    argparser.add_argument('--u_std', default=None, type=float, help='standard deviation of u')

    argparser.add_argument('--smile_tau', default=None, type=float, help='smile estimator clip val')
    argparser.add_argument('--D', default=None, type=float, help='')
    argparser.add_argument('--P', default=None, type=float, help='')
    argparser.add_argument('--gaussian_r', default=None, type=float, help='')

    argparser.add_argument('--bern_p', default=None, type=float, help='')

    argparser.add_argument('--gamma_p', default=None, type=float, help='')
    argparser.add_argument('--gamma_cap', default=None, type=float, help='')
    argparser.add_argument('--increase_gamma', default=None, type=int, help='')
    argparser.add_argument('--perception', default=None, type=int, help='')
    argparser.add_argument('--max_gamma', default=None, type=int, help='')
    argparser.add_argument('--latent_dim', default=None, type=int, help='')

    argparser.add_argument('--batch_size', default=None, type=int, help='')
    argparser.add_argument('--num_iter', default=None, type=int, help='')
    argparser.add_argument('--eval_batches', default=None, type=int, help='')
    argparser.add_argument('--lr', default=None, type=float, help='')
    argparser.add_argument('--num_epochs', default=None, type=int, help='')
    argparser.add_argument('--regularize', default=None, type=int, help='')
    argparser.add_argument('--quantize_alphabet', default=None, type=int, help='')

    argparser.add_argument('--critic_hidden_dim', default=None, type=int, help='')
    argparser.add_argument('--critic_embed_dim', default=None, type=int, help='')
    argparser.add_argument('--critic_layers', default=None, type=int, help='')
    argparser.add_argument('--cuda_visible', default=None, type=int, help='')
    argparser.add_argument('--using_wandb', default=None, type=int, help='')
    argparser.add_argument('--critic_activation', default=None, type=str, help='')

    argparser.add_argument('--save_epoch', default=None, type=int, help='')

    argparser.add_argument('--clip_grads', default=None, type=int, help='')
    argparser.add_argument('--grad_clip_val', default=None, type=float, help='')

    argparser.add_argument('--kl_loss', default=None, type=str,
                           help='(smile, infonce, js, dv, tuba, nwj, ent_decomp)')
    argparser.add_argument('--wandb_project_name', default=None, type=str, help='wandb project name')
    argparser.add_argument('--quiet', dest='quiet', action='store_true')

    argparser.add_argument('--xy_dim', default=None, type=int, help='')


    argparser.set_defaults(quiet=False)

    args = argparser.parse_args()
    return args



