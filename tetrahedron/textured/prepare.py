import cv2
import numpy as np


def prepare_texture(size, in_files, out_file):
    images = [cv2.resize(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB), size) for file in in_files]
    texture = np.concatenate(images, axis=1)
    cv2.imwrite(out_file, texture)

size = (500, 500)
in_files_1 = ["./image/image_1.png",
              "./image/image_2.png",
              "./image/image_3.png",
              "./image/image_4.png",]
out_file_1 = "./image/texture.png"

prepare_texture(size, in_files_1, out_file_1)