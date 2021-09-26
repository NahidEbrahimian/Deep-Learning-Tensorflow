
import numpy as np
import cv2
import matplotlib.pyplot as plt
from retinaface import RetinaFace
import argparse
from keras.models import load_model
from config import *

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str)
args = parser.parse_args()

model = load_model('model.h5')

faces = RetinaFace.extract_faces(img_path = args.image_path, align = True)
image = faces[0]
image =cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (width, height))

plt.imshow(image)

image = image/255.0
image = np.expand_dims(image, axis=0)

age = model.predict(image)
print('Age predicted: ', age[0])