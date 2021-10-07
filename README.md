# Deep-Learning-Course

## 01- CNN and MLP

- Test accuracy comparison between MLP and Deep Learning

| Dataset | MLP (Machine Learning) | CNN + MLP (Deep Learning) |
| :---         |     :---:      |          :---: |
| Mnist  | 0.97     | 0.98    |
|Fashion Mnist     | 0.86       | 0.89    |
|Cfar 10     | 0.37       | 0.63      |
|Cfar 100     | 0.13       | 0.28      |

#

## 02- Sheikh-Recognition

- Train Neural Network for normal person and sheikh images classification

- [x] train.ipynb

- [x] inference.py

#### Dataset: 

contain images from two classes, normal person and sheikh

For inference run the following command:

```
!python3 inference.py --input_image test/image3.jpg

```

#

## 03- Sheikh-Recognition-bot

- Train Neural Network for normal person and sheikh images classification

- [x] train.ipynb

- [x] bot.py

#### Dataset:

contain images from two classes, normal person and sheikh

#### Usage:

1. Click [here](https://t.me/SheikhRecognition_bot) to open the chat with the bot in the Telegram app

2. Start the bot and send him a photo

## 04- Persian Face Recognition

- [ ] train.ipynb

- [x] preprocess.py

- [ ] inference.py

#### Preprocessing

Preprocess stage consists of 4 common stages: detect, align, represent and verify. link: [Github]( https://github.com/serengil/retinaface#:~:text=RetinaFace%20is%20a%20deep%20learning,is%20published%20by%20Stanislas%20Bertrand.)

1. You must first install retinaface:

```
!pip install retina-face
```

2. Run the following command to apply preprocessing:

```
!python3 preprocess.py --input_images_dir "./input_images" --output_dir "./output_dir"
```

#

## 05- 17Flowers Classification

#### Dataset:

Contain images from 17 classes of flowers in two subset, train and test.

Dataset link: [Flowers]( https://drive.google.com/drive/folders/1-7GcWubgmhIImiZUrghV3haBjIeLoEhf?usp=sharing)

#### Result:

Comparison accuracy of pretrained models that used in transfer learning on test data:

| Model | Accuracy |
| :---         |     :---:      |
| Vgg16  | 0.67     | 
|Vgg19     | 0.70       | 
|ResNet50V2    | 0.82       |
|MobileNetV2     | 0.37       | 

#

## 07- UTKFace-Age prediction-Regression

- Train Neural Network on UTKFace dataset using tensorflow and keras

- [x] train.ipynb

- [x] inference.py

#### Dataset:

Dataset link: [UTKFace-dataset]( https://www.kaggle.com/jangedoo/utkface-new)


#### inference:

1- First install retina-face
```
!pip install retina-face
```

2- Run the following command:

```
!python3 inference.py --image_path 'input/08.jpg'
```

## 08- Gender Recognition

- Train Neural Network on gender-recognition dataset using tensorflow and keras

- [x] train.ipynb

- [ ] inference.py

#### Dataset:

Dataset link: [celeba-dataset]( https://www.kaggle.com/ashishjangra27/gender-recognition-200k-images-celeba)


## 10- Houses Price Image Regression

- Train Neural Network for house price prediction using images

- [x] train.ipynb

- [x] inference.py

Reference: [KerasRegressionandCNNs]( https://www.pyimagesearch.com/2019/01/28/keras-regression-and-cnns/)

#

## 11- DCGAN_on_Mnist_dataset

- Training DCGAN on Mnist dataset

![DCGAN-02](https://user-images.githubusercontent.com/82975802/135684094-e46ae7fd-4d25-401c-b91b-7fe30cef7164.gif)

Reference: [tensorflow-tutorials]( https://www.tensorflow.org/tutorials/generative/dcgan)

## 12- DCGAN_on_Celeb_A_dataset

- Face Generator, Training DCGAN on celeba dataset

![DCGAN-Celeb_A-02 (1)](https://user-images.githubusercontent.com/82975802/135740954-117c825d-d07e-4b8f-a100-74d61a72b933.gif)

Dataset link: [celeba-dataset]( https://www.kaggle.com/ashishjangra27/gender-recognition-200k-images-celeba)

