# Face Recognition

Face Recognition on Image using tensorflow and keras




| ![angelina-jolie](https://user-images.githubusercontent.com/82975802/137978311-bd050d06-7ad4-4a0d-b115-09aa607327b4.jpg) | ![queen-elizabet](https://user-images.githubusercontent.com/82975802/137979290-a87714cf-f086-4258-a02a-99b5c97d4147.jpg)|
| :---:         |     :---:      |
| ![Scarlett-Johansson](https://user-images.githubusercontent.com/82975802/137978399-8b11247a-7357-4215-80d6-db41f52b9bbf.jpg) | ![behnam-bani](https://user-images.githubusercontent.com/82975802/137979087-157d0c5d-65b2-400a-a647-7ecd75a167d7.jpg)| 




- Train Neural Network on 7-7 dataset using tensorflow and keras

- [x] train.ipynb

- [x] inference-img.py

- [ ] train.py

#

### Dataset:

Dataset contain images in 14 classes

Dataset link: [7-7 dataset]( https://drive.google.com/drive/folders/1WGSotRtFPYGuxPEGkWWRsBPlVXFSvl7p?usp=sharing)

#

### Train:

For training you can run `FaceRecognition_ExpertMode.ipynb` 

```
```

#

### Inference:


First download model from this link: [MyModel](https://drive.google.com/file/d/1akU4IbNxq0JnarxF_SrL63TQaGfIA7b_/view?usp=sharing)

Then, put your images in `./input` directory and run the following command. output images will be saved on `output` folder.

```
python3 inference-img.py --input input/queen-elizabet.jpg

```

#

### Useful links:

Face-Alignment preprocessing used in the inference step: https://github.com/SajjadAemmi/Face-Alignment

