# -*- coding: utf-8 -*-


class Config(object):
    # path
    src_path = "data"
    dst_path = "results"
    val_path = "validation"
    log_path = "logs/"
    model_path = "checkpoints"

    # training parameters
    gpu = False
    img_sz = 64
    batch_sz = 64
    n_f = 64
    n_feat_dim = 2048
    n_hidden_dim = 512
    lr = 1e-4
    beta1 = 0.5
    beta2 = 0.999
    wd = 1e-4
    max_epoch = 150
    start_mix = 30
    end_mix = 70
    lambda_topo = 2
    lambda_metric = 0.5
    save_freq = 10


opt = Config()
