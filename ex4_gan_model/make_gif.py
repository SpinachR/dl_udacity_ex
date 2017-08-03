import numpy as np
import os
from PIL import Image, ImageSequence
from images2gif import writeGif
from utils import get_image
from utils import read_image_list

def get_shape_for_data(filenames):
    array = [Image.open(batch_file) for batch_file in filenames]
    return array


# get the numpy array of images from the path from image
def get_image(image_path):
    # Get the images from the path of image
    list_file = read_image_list(image_path)
    list_file.sort(compare)

    image_array = get_shape_for_data(list_file)

    return image_array


def compare(x , y):
    stat_x = os.stat(x)
    stat_y = os.stat(y)
    if stat_x.st_ctime < stat_y.st_ctime:
        return -1
    elif stat_x.st_ctime > stat_y.st_ctime:
        return 1
    else:
        return 0


def make_gif(images):
    writeGif('result.gif' , images , duration=0.5)
    #Run
    image_path = './gif_images/'
    image_array = get_image(image_path)
    make_gif(image_array)