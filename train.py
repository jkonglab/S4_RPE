# -*- coding: utf-8 -*-

import cv2
import time
import torch
import random
import torch.nn as nn
import torchvision as tv
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import *
from model import *
from config import opt
from glob import glob


def train(**kwargs):
    """Training Network"""

    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device = torch.device("cuda:1" if opt.gpu else "cpu")
    setattr(opt, "device", device)

    # 1. Load data
    transforms_com = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.RandomVerticalFlip(),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.RandomCrop(opt.img_sz)
    ])

    transforms_sep = tv.transforms.Compose([
        randomGaussianNoise(mean=0, std=0.5, p=0.7),
        tv.transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5)),
        tv.transforms.ColorJitter(brightness=[0.2, 0.7])
    ])

    img_norm = tv.transforms.Normalize(0.5, 0.5)
    img_rec = tv.transforms.Normalize(-1, 2)

    dataloader = DataLoader(
        ImageDataset(opt.src_path, transforms_com, transforms_sep, random_erase=0),
        batch_size=opt.batch_sz,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )

    # 2. Initialize network
    encoder = Encoder(n_f=opt.n_f, n_downsample=3, n_res=6, in_ch=1, feat_idx=[6], device=device)
    decoder = Decoder(n_f=opt.n_f, n_downsample=3, out_ch=1, device=device)
    projector = MultiLayerPerceptron(in_dim=encoder.get_feat_dim(opt.img_sz**2)[0], out_dim=opt.n_feat_dim,
        out_norm=True, hidden_dim=opt.n_hidden_dim, n_hidden=2, device=device)
    predictor = MultiLayerPerceptron(in_dim=opt.n_feat_dim, out_dim=opt.n_feat_dim, out_norm=False,
        hidden_dim=opt.n_hidden_dim, n_hidden=1, device=device)

    # 3. Define Optimizing Strategy
    params = [
        {"params": encoder.parameters()},
        {"params": decoder.parameters()},
        {"params": projector.parameters()},
        {"params": predictor.parameters()},
    ]
    optimizer = torch.optim.Adam(
        params,
        betas=(opt.beta1, opt.beta2),
        lr=opt.lr,
        weight_decay=opt.wd
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2
    )
    criterion_ssl = nn.CosineSimilarity().to(device)
    criterion_idt = nn.L1Loss().to(device)
    criterion_topo = TopoLoss(device)

    # 4. Create Valdation Set
    val_pos_path = os.path.join(opt.val_path, "positive/*.*")
    tensors = []
    for file_name in glob(val_pos_path):
        tmp_img = cv2.imread(file_name)
        tmp_tensor = img_norm(transforms_com(tmp_img[:, :, 1]))
        tensors.append(tmp_tensor)
    val_real_pos = torch.stack(tensors).to(device)

    val_neg_path = os.path.join(opt.val_path, "negative/*.*")
    tensors = []
    for file_name in glob(val_neg_path):
        tmp_img = cv2.imread(file_name)
        tmp_tensor = img_norm(transforms_com(tmp_img[:, :, 1]))
        tensors.append(tmp_tensor)
    val_real_neg = torch.stack(tensors).to(device)

    # 5.Train Networks
    log_path = opt.log_path + "Simsiam-CutMix-Topo_" + \
               time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    writer = SummaryWriter(log_path)

    for epoch in range(opt.max_epoch):
        # Train
        encoder.train()
        decoder.train()
        projector.train()
        predictor.train()

        for i, batch in enumerate(dataloader):
            x = batch["orig"].to(device)
            x1 = batch["A"].to(device)
            x2 = batch["B"].to(device)

            # Forward
            y1 = encoder(x1)[0]
            y2 = encoder(x2)[0]
            z1 = projector(y1.view(opt.batch_sz, -1))
            z2 = projector(y2.view(opt.batch_sz, -1))
            p1 = predictor(z1)
            p2 = predictor(z2)

            w1 = decoder(y1)
            w2 = decoder(y2)

            # Backpropagation
            optimizer.zero_grad()

            loss_input_metric = -0.5 * (criterion_ssl(p1, z2.detach()).mean() + criterion_ssl(p2, z1.detach()).mean())
            loss_metric = loss_input_metric

            loss_idt_ref = (criterion_idt(x, w1) + criterion_idt(x, w2)) * 0.5
            loss_idt_mutual = criterion_idt(w1, w2)
            loss_idt = mix_loss(loss_idt_mutual, loss_idt_ref, epoch, opt.start_mix, opt.end_mix, 0, 0.5)


            loss_topo = (criterion_topo(w1) + criterion_topo(w2)) * 0.5 if epoch > opt.start_mix else 0
            loss_topo = mix_loss(loss_topo, 0, epoch, opt.start_mix, opt.end_mix, 0, 1)

            loss = loss_idt * 2 + loss_metric * opt.lambda_metric + loss_topo * opt.lambda_topo
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validate
        encoder.eval()
        decoder.eval()
        projector.eval()
        predictor.eval()

        val_fake_pos = decoder(encoder(val_real_pos)[0])
        val_fake_neg = decoder(encoder(val_real_neg)[0])
        _, topo_ref = criterion_topo(val_fake_neg, output_ref=True)

        writer.add_scalar("loss_input_metric", loss_input_metric.item(), epoch)
        writer.add_scalar("loss_idt", loss_idt.item(), epoch)
        writer.add_scalar("loss_topo", loss_topo.item() if epoch > opt.start_mix else 0, epoch)

        cur_time = time.strftime('%m-%d %H:%M:%S', time.localtime(time.time()))
        print("epoch:", epoch, cur_time)

        # Save models
        if (epoch + 1) % opt.save_freq == 0:
            torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict()},
                        "%s/models_%s.pth"% (opt.model_path, epoch+1))
            writer.add_image("fake_pos/%d" % (epoch+1), tv.utils.make_grid(img_rec(val_fake_pos), 4), epoch)
            writer.add_image("fake_neg/%d" % (epoch+1), tv.utils.make_grid(img_rec(val_fake_neg), 4), epoch)
            writer.add_image("topo_ref/%d" % (epoch+1), tv.utils.make_grid(topo_ref, 4), epoch)

    writer.close()


if __name__ == "__main__":
    train(gpu=True)
