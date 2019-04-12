# VLocNet
This repository is a PyTorch implementation of [*Deep Auxiliary Learning for Visual Localization and Odometry*](https://arxiv.org/abs/1803.03642), known as VLocNet.

This is **NOT** an official version published by the authors, but a self-implemented project (with some codes referring to the project of [MapNet](https://github.com/NVlabs/geomapnet)). 
It should be mentioned that the current performance is still **subpar** compared with the reported by the paper, although our best have been tried, but **close to** the [MapNet](https://github.com/NVlabs/geomapnet), on [7Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) dataset.  

We release this code mainly for two purpose:
+ Providing a bedrock for those who also endeavor to implement this work 
+ Encouraging the guys interested in our work to find the bugs if existing and explore the tricks before the code released by the authors. 

*Feel free to create new issues about our work, and share your ideas and suggestions*. 

## Setup

1. Install miniconda with Python 3.5+.
2. Create the mapnet Conda environment: conda env create -f environment.yml.
3. Activate the environment: conda activate mapnet_release.

## Data
We only support the [7Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) dataset for now. The datasets live in the `datasets` directory. We provide skeletons with symlinks to get you started. Let us call your 7Scenes download directory 7SCENES_DIR. You will need to make the following symlinks:

`cd datasets && ln -s 7SCENES_DIR 7Scenes` 

## Running the code
The python training script and test script are included in `train.py` and `test.py` separately. The configuration is straightforward and for simplicity some `bool` options are replaced by `int` ( 0 for `False` and 1 for `True`).  
The training and testing can be carried out by the commands 
`python train.py` and `python test.py`.
The specific configurations are at your discretion.  