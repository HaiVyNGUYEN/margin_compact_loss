# Large Margin Discriminative Loss for Classification

This Repo contains all the files related for computing Lens Depth and for conducting experiments in the related paper.

## Dependencies

The code is implemented based mainly on python library Pytorch (and torchvision). All needed libraries can be found in  [requirements.txt](https://github.com/HaiVyNGUYEN/margin_compact_loss/blob/master/requirements.txt). The code is supposed to be run in Linux but can be easily adapted for other systems. We strongly recommend to create virtual environment for a proper running (such as conda virtual env). This can be easily done in linux terminal as follow:
```
conda create -n yourenvname python=x.x anaconda
```
Then, to activate this virtual env:
```
conda activate yourenvname
```
To install a package in this virtual env:
```
conda install -n yourenvname [package]
```

To quit this env:

```
conda deactivate
```

## Data

In this work, [SVHN](http://ufldl.stanford.edu/housenumbers/), [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html). All these datasets can be downloaded using standard Deep Learning library [Pytorch](https://pytorch.org/). As the training schme is completely the same for CIFAR10 and SVHN, here we present only CIFAR10, but you can easily load SVHN dataset and apply this code.

## Running

We have following files:

1. [download_data.py](https://github.com/HaiVyNGUYEN/margin_compact_loss/blob/master/download_data.py): run this code snippet for downloading CIFAR10.
2. [archi.py](https://github.com/HaiVyNGUYEN/margin_compact_loss/blob/master/archi.py): defining ResNet18 model.
3. [training_utils.py](https://github.com/HaiVyNGUYEN/margin_compact_loss/blob/master/training_utils.py): necessary tools for training and testing.
4. [train_test_softmax.py](https://github.com/HaiVyNGUYEN/margin_compact_loss/blob/master/train_test_softmax.py): train and test model with softmax loss.
5. [train_test_discriminative_loss.py](https://github.com/HaiVyNGUYEN/margin_compact_loss/blob/master/train_test_discriminative_loss.py): train and test model with our loss.
6. [qualitative_results.ipynb](https://github.com/HaiVyNGUYEN/margin_compact_loss/blob/master/qualitative_results.ipynb): jupyter notebook for qualitative results using t-SNE.


Note that to run a file xxx.py, you need to to activate your env (as mentionned above) and then on your terminal, type

```
python xxx.py
```


![Alt text](https://github.com/HaiVyNGUYEN/margin_compact_loss/blob/master/image/t_sne.png)

