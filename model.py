import torch.nn as nn
import torch
from utils import *


class Encoder(nn.Module):

    def __init__(self, n_f, n_downsample, n_res, in_ch, feat_idx=None, device=torch.device("cpu")):
        super().__init__()

        model = [ConvBlock(in_ch, n_f, 7, padding=3, padding_mode="reflect")]
        for i in range(n_downsample):
            factor = 2 ** i
            model += [ConvBlock(n_f*factor, n_f*factor*2, 3, stride=2, padding=1, padding_mode="reflect")]
        factor = 2 ** n_downsample
        for i in range(n_res):
            model += [ResBlockV2(n_f*factor, n_f*factor)]

        self.encoder = nn.Sequential(*model)
        self.encoder.to(device)
        self.encoder.apply(init_weights)

        self.n_f = n_f
        self.n_downsample = n_downsample
        self.n_res = n_res
        self.feat_idx = feat_idx if feat_idx else list(range(self.n_downsample+1))

    def get_feat_dim(self, input_size):
        base_dim = input_size * self.n_f
        feat_dim = []
        for i in self.feat_idx:
            if i > self.n_downsample:
                feat_dim.append(base_dim//(2**self.n_downsample))
            else:
                feat_dim.append(base_dim//(2**i))
        return feat_dim

    def forward(self, x):
        feat = x
        feats = []
        for i, layer in enumerate(self.encoder):
            feat = layer(feat)
            if i in self.feat_idx:
                feats.append(feat)
        return feat, feats


class Decoder(nn.Module):

    def __init__(self, n_f, n_downsample, out_ch, device=torch.device("cpu")):
        super().__init__()

        model = []
        for i in range(n_downsample):
            factor = 2 ** (n_downsample - i - 1)
            model += [UpConvBlock(n_f*factor*2, n_f*factor)]
        model += [nn.Conv2d(n_f, out_ch, 7, padding=3, padding_mode="reflect"), nn.Tanh()]

        self.decoder = nn.Sequential(*model)
        self.decoder.to(device)
        self.decoder.apply(init_weights)

    def forward(self, x):
        return self.decoder(x)


class MultiLayerPerceptron(nn.Module):

    def __init__(self, in_dim, out_dim, out_norm, hidden_dim, n_hidden, device=torch.device("cpu")):
        super().__init__()

        model = [nn.Linear(in_dim, hidden_dim, bias=False),
                 nn.BatchNorm1d(hidden_dim),
                 nn.ReLU(inplace=True)]

        for i in range(n_hidden-1):
            model += [nn.Linear(hidden_dim, hidden_dim, bias=False),
                      nn.BatchNorm1d(hidden_dim),
                      nn.ReLU(inplace=True)]

        model += [nn.Linear(hidden_dim, out_dim, bias=False)]
        if out_norm:
            model += [nn.BatchNorm1d(out_dim)]

        self.mlp = nn.Sequential(*model)
        self.mlp.to(device)
        self.mlp.apply(init_weights)

    def forward(self, x):
        return self.mlp(x)
