import numpy as np
import torch
import os
# from training_routines import GetTrainer
# from model import GetModel
# from data import GetDataGen
# from utilities import PreprocessMeta

def main():
    print('mained')
    np.random.seed(16)
    print('l')
    # torch.manual_seed(16)
    #
    # config = PreprocessMeta()
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda_visible)
    # # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = '0:4096'
    # if not(hasattr(config, 'quiet') and config.quiet):
    #     print(config)
    #
    # data_gen = GetDataGen(config)
    # model = GetModel(config)
    # trainer = GetTrainer(config, model, data_gen)
    #
    # trainer.train()
    # trainer.process_results()




if __name__ == '__main__':
    main()