import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from config import *
from keras import Input


class MyModel(tf.keras.Model):
  def __init__(self, num_class, input_shape):
    super().__init__()
    self.conv1 = Conv2D(32, (3, 3), activation='relu', input_shape = input_shape)
    self.conv2 = Conv2D(64, (3, 3), activation='relu')
    self.conv3 = Conv2D(128, (3, 3), activation='relu')
    self.conv4 = Conv2D(256, (3, 3), activation='relu')
    self.conv5 = Conv2D(128, (3, 3), activation='relu')
    self.maxpool = MaxPooling2D()
    self.flatten = Flatten()
    self.dropout = Dropout(0.2)
    self.fc1 = Dense(128, activation='relu')
    self.fc2 = Dense(64, activation='relu')
    self.fc3= Dense(num_class, activation='softmax')
    self.dim = input_shape


  def call(self, x, *args, **kwargs):
    conv1 = self.conv1(x)
    maxpool1 = self.maxpool(conv1)

    conv2 = self.conv2(maxpool1)
    maxpool2 = self.maxpool(conv2)

    conv3 = self.conv3(maxpool2)
    maxpool3 = self.maxpool(conv3)

    conv4 = self.conv4(maxpool3)
    maxpool4 = self.maxpool(conv4)


    flatten = self.flatten(maxpool4)
    dropout = self.dropout(flatten)
    fc1 = self.fc1(dropout)
    dropout = self.dropout(fc1)
    fc2 = self.fc2(dropout)
    dropout = self.dropout(fc2)
    output = self.fc3(dropout)

    return output

  def build_graph(self):
      x = Input(shape=(self.dim))
      return tf.keras.Model(inputs=[x], outputs=self.call(x))

