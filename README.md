# Correlation-Aware Mutual Learning for Semi-supervised Medical Image Segmentation

### Introduction
This repository is for our paper: 'Correlation-Aware Mutual Learning for Semi-supervised Medical Image Segmentation'.

### Requirements
This repository is based on PyTorch 1.8.1, CUDA 10.2 and Python 3.6.9; All experiments in our paper were conducted on a single NVIDIA TITAN RTX GPU.

### Usage
1. Clone the repo.;
```
https://github.com/Herschel555/CAML.git
```
2. Put the data in './CAML/data';

3. Train the model;
```
cd CAML
# e.g., for 5% labels on LA
python code/train_caml.py --labelnum 4 --gpu 0 --seed 1337
```
4. Test the model;
```
cd CAML
# e.g., for 5% labels on LA
python code/test_3d.py --labelnum 4 --gpu 0 --seed 1337
```


### Acknowledgements:
Our code is origin from [UAMT](https://github.com/yulequan/UA-MT), [SSL4MIS](https://github.com/HiLab-git/SSL4MIS) and [MC-Net](https://https://github.com/ycwu1997/MC-Net). Thanks for these authors for their valuable works.
