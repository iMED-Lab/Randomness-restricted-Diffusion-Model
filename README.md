# R²diff: Randomness-restricted Diffusion Model for Ocular Surface Structure Segmentation

A General Method for Ocular Surface Segmentation Based on Diffusion Models. We will keep this code updated continuously.

### Requirements

1. System Requirements:
	- NVIDIA GPUs, CUDA supported.
	- Ubuntu 20.04 workstation or server
	- Anaconda environment
	- Python 3.8
	- PyTorch 1.12 
	- Git

2. Installation:
   - `git clone https://github.com/iMED-Lab/Randomness-restricted-Diffusion-Model.git`
   - `cd ./Randomness-restricted-Diffusion-Model`
   - `conda env create -f environment.yaml`
   - `conda activate rrdm_env`

## Training&Sampling
1. If you want to train a new model, please run the following code:
```
python scripts/segmentation_train.py
```
2. After training, you can generate a mask like so:
```
python scripts/segmentation_sample.py
```

3. If you want to train on your own dataset:
	- Refer to guided_diffusion/dataset.py. You only need to modify the path and ensure that different classes in the labels are assigned different pixel values (0-255).

## Pre-trained weights 
1. Here, the model weights obtained from training on public datasets are provided:
	- link：https://pan.baidu.com/s/1h_cxECqzqkfN5O7M5_aVBA 
	- password：imed 

### License
MIT License
