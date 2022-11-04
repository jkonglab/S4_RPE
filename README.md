# S<sup>4</sup>
We provide our PyTorch implementation of our S<sup>4</sup> model. The developed method trains a deep neural network with unlabeled images. Specifically, a reconstruction loss by the encoder-decoder component and a pairwise representation loss by auxiliary MLP heads are used to extract structural information of cell borders. Additionally, a morphological loss guides the network to output binary segmentation results.

The model presents better performance compared with some SOTA approaches such as UNet, DeepLab, MultiResUNet, and[CellPose](https://www.cellpose.org/) in the RPE cell border segmentation task.
***
### Prerequisites
* CPU or NVIDIA GPU
* Linux or macOS
* Python 3.10
* PyTorch 1.8.1
***
### Usage
* Train model:
```
python train.py
```
* Test model:
```
python test.py
```
### Note
* `model.py` constructs the networks including the encoder, decoder, and MLPs in our developed model.
* `utils.py` defines modules to build the network, loss functions, etc.
* In `config.py` users can change configurations including I/O paths, filter size and number for the first layer of the network, and traning hyperparameters. More explanations are given in the file.
