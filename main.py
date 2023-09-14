import numpy as np
import torch
import os
from training_routines import GetTrainer
from model import GetModel
from data import GetDataGen
from utilities import PreprocessMeta

def main():
    np.random.seed(16)
    torch.manual_seed(16)

    config = PreprocessMeta()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda_visible)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    if hasattr(config, 'xy_dim'):
        config.y_dim = config.x_dim
        config.u_dim = config.x_dim

    if not(hasattr(config, 'quiet') and config.quiet):
        print(config)



    data_gen = GetDataGen(config)
    model = GetModel(config)
    trainer = GetTrainer(config, model, data_gen)

    trainer.train()
    # trainer.process_results()




if __name__ == '__main__':
    main()