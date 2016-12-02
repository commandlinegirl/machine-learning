from scipy import ndimage
from PIL import Image

def read_img(img_no):
    """reads in image as an array"""
    return ndimage.imread(str(img_no) + '.jpg')

def save_img(name, img_array, dir):
    """saves image"""
    img = Image.fromarray(img_array)
    img.save(dir + str(name) + '.jpg')

