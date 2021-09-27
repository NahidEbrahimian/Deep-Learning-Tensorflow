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

#### Preprocessing

Preprocess stage consists of 4 common stages: detect, align, represent and verify.

1. you must first clone retinaface using the following command:

```
!git clone https://github.com/serengil/retinaface
```

2. copy `preprocess.py` in `./retinaface` directory:

```
!cp preprocess.py ./retinaface
```

3. then, run the following command to apply preprocessing:

```
%cd ./retinaface
!python3 preprocess.py --input_images_dir "./dataset_dir" --output_dir "./output_dir"
```

#

## 05- 17Flowers Classification

#### Dataset:

Contain images from 17 classes of flowers in two subset, train and test.

Dataset: [Flowers]( https://drive.google.com/drive/folders/1-7GcWubgmhIImiZUrghV3haBjIeLoEhf?usp=sharing)

#### Result:

Comparison accuracy of pretrained models that used in transfer learning on test data:

| Model | Accuracy |
| :---         |     :---:      |
| Vgg16  | 0.67     | 
|Vgg19     | 0.70       | 
|ResNet50V2    | 0.82       |
|MobileNetV2     | 0.37       | 

#

## 06- WeatherPrediction_Regression

- Train Neural Network on weather-dataset using tensorflow and keras

#### Dataset:

Dataset: [weather-dataset]( https://drive.google.com/drive/folders/10OdTbgLI8O-ZezfHopbpbqgJ_lI9M5D-?usp=sharing)

- Loss on test data: 3.0455

## 07- UTKFace-Age prediction-Regression

- Train Neural Network on UTKFace dataset using tensorflow and keras

- [x] train.ipynb

- [x] inference.py

#### Dataset:

Dataset: [UTKFace-dataset]( https://www.kaggle.com/jangedoo/utkface-new)


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

Dataset: [gender-recognition-dataset]( https://www.kaggle.com/ashishjangra27/gender-recognition-200k-images-celeba)

## 09- Face Mask Detection

- Train Neural Network on face-mask dataset using tensorflow and keras

- [x] train.ipynb

- [ ] inference.py

#### Dataset:

Dataset: [face-mask-dataset]( ashishjangra27/face-mask-12k-images-dataset)



