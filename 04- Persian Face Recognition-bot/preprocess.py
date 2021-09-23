import os
import cv2
from retinaface import RetinaFace
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--input_images_dir", type=str)
parser.add_argument("--output_dir", type=str)
args = parser.parse_args()

list_of_images = os.listdir(args.input_images_dir)

for i, im in enumerate(list_of_images):
  faces = RetinaFace.extract_faces(img_path = os.path.join(args.input_images_dir, im), align = True)

  for face in faces:
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(args.output_dir, f'{i}.jpg'), face)