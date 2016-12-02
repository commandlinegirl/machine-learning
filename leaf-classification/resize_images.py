import gzip, binascii, struct, glob
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from scipy import ndimage
from PIL import Image
import inout

IMAGE_DIR = './images/'
IMAGE_SCALED_TEST_DIR = './images_scaled/test/'
IMAGE_SCALED_TRAIN_DIR = './images_scaled/train/'
IMAGE_NUM = 1584
PIXEL_DEPTH = 255
NUM_LABELS = 10

def create_image(w, h, pixel):
    """returns an array of a given shape and value"""
    return np.full((w, h), pixel, dtype=np.uint8)

def superpose_array(smaller, larger): 
    for i in range(smaller.shape[0]):
        for j in range(smaller.shape[1]):
            larger[i][j] = smaller[i][j]         
    return larger

def max_dims():
    w = h = 0
    for i in range(1, IMAGE_NUM + 1):
        shape = inout.read_img(i).shape
        if shape[0] > w:
            w = shape[0]
        if shape[1] > h:
            h = shape[1]
    return (w, h)

def get_img_numbers(filename):
    train_imgs = []
    with open(filename) as f:
        f.readline()
        for line in f:
            train_imgs.append(int(line.split(',')[0]))
    return set(train_imgs)    

#### Run

train_nums = get_img_numbers('train.csv') 
test_nums = get_img_numbers('test.csv') 

print 'Count of images in train: ' + str(len(train_nums))
print 'Count of images in test: ' + str(len(test_nums))

w, h = max_dims()
for i in range(1, IMAGE_NUM + 1):
    print i
    one_sized_img = superpose_array(inout.read_img(i), create_image(w, h, 0))
   
    if i in train_nums:
        inout.save_img(i, one_sized_img, IMAGE_SCALED_TRAIN_DIR)
    else:
        inout.save_img(i, one_sized_img, IMAGE_SCALED_TEST_DIR)
 
assert len(glob.glob(IMAGE_SCALED_TRAIN_DIR + "*.jpg")) == len(train_nums)
assert len(glob.glob(IMAGE_SCALED_TEST_DIR + "*.jpg")) == len(test_nums)

print 'Done'
