
import os
import numpy as np
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str)
args = parser.parse_args()

model = load_model("weights/model.h5")

images = []
outputImage = np.zeros((64, 64, 3), dtype="uint8")

for im in os.listdir(args.input_dir):
  image = cv2.imread(os.path.join(args.input_dir, im))
  image = cv2.resize(image, (32, 32))
  images.append(image)

outputImage[0:32, 0:32] = images[0]
outputImage[0:32, 32:64] = images[1]
outputImage[32:64, 32:64] = images[2]
outputImage[32:64, 0:32] = images[3]

outputImage = np.array(outputImage)
outputImage = outputImage / 255

outputImage = outputImage.reshape(1, 64, 64, 3)
pred = model.predict([outputImage])
print('Predicted Price: ', pred[0])