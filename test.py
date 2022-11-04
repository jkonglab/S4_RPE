import torch
import numpy as np
import torch.nn as nn
import torchvision as tv
from utils import *
from model import *
from config import opt
from torchstat import stat
from skimage.filters import frangi
from datetime import datetime, timedelta
from skimage.filters import threshold_local


@torch.no_grad()
def predict(model_path, test_path, dst_path, transforms):

    device = torch.device("cpu")
    params = torch.load(model_path, map_location=device)

    encoder = Encoder(n_f=opt.n_f, n_downsample=3, n_res=6, in_ch=1, feat_idx=[6], device=device).eval()
    encoder.load_state_dict(params["encoder"])
    decoder = Decoder(n_f=opt.n_f, n_downsample=3, out_ch=1, device=device).eval()
    decoder.load_state_dict(params["decoder"])
    criterion_topo = TopoLoss(device)

    for path in glob(os.path.join(test_path, '*.tif')):
        name = os.path.basename(path)
        print(name)
        img = cv2.imread(path)
        img = img[:, :, 1]
        height, width = img.shape
        height_pad = (8 - height % 8) % 8
        width_pad = (8 - width % 8) % 8
        img = cv2.copyMakeBorder(img, 0, height_pad, 0, width_pad, cv2.BORDER_CONSTANT, value=0)
        X = transforms(img).to(device)
        X.unsqueeze_(0)
        Y = decoder(encoder(X)[0])

        _, Y_topo = criterion_topo(Y, output_ref=True)
        output = Y_topo.squeeze().cpu().numpy() * 255
        output = output[0:height, 0:width]
        cv2.imwrite(os.path.join(dst_path, name), output.astype(np.uint8))



if __name__ == "__main__":
    transforms = tv.transforms.Compose([
        tv.transforms.ToPILImage(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(0.5, 0.5)
    ])

    model_path = "checkpoints/models_100.pth"
    test_path = "test"
    dst_path = "results"
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    predict(model_path, test_path, dst_path, transforms)
