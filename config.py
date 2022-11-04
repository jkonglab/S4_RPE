# -*- coding: utf-8 -*-


class Config(object):
    # path
    src_path = "data"           # path to load training data
    log_path = "logs/"          # path to save training logs
    model_path = "checkpoints"  # path to save (intermediate) models

    # training parameters
    gpu = False         # whether to train with GPU
    img_sz = 64         # image size
    batch_sz = 64       # batch size
    n_f = 64            # number of filters at the highest level
    n_feat_dim = 2048   # length of representation vector
    n_hidden_dim = 512  # number of nodes in MLP hidden layers
    lr = 1e-4           # learning rate
    beta1 = 0.5         # parameter 1 for Adam optimizer
    beta2 = 0.999       # parameter 2 for Adam optimizer
    wd = 1e-4           # weight decay rate
    max_epoch = 150     # max epoch number
    start_mix = 30      # start time of training strategy transition
    end_mix = 70        # end time of training strategy transition
    lambda_topo = 2     # weight factor for topology loss
    lambda_metric = 0.5 # weight factor for pairwise learning loss
    save_freq = 25      # model saving frequency


opt = Config()
