## Houses Price Image Regression

- Train Neural Network for house price prediction using images

#### Dataset:

A total of 535 houses are included in the dataset, and 2,140 total images. four images of each house: bedroom, bathroom, Kitchen, frontal view of the house.

#

#### Train:

1. Download dataset uising the following command:

```
!git clone https://github.com/emanhamed/Houses-dataset
```

2. Run this command for training:

```
!python3 cnn_regression.py --dataset /content/Houses-dataset/Houses\ Dataset
```
#

#### Inference:

Put four images of house in `input` folder and run the following command.

```
!python3 inference.py --input_dir ./input
```

#

**Reference** : [KerasRegressionandCNNs]( https://www.pyimagesearch.com/2019/01/28/keras-regression-and-cnns/)
