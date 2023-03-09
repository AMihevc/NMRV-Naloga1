import cv2
import matplotlib.pyplot as plot
import numpy as np

from ex1_utils import *
from lucas_Kanade import lucaskanade

#function to plot the images
def plot_images(img1, img2, cmap=None):
    plot.subplot(1,2,1)
    plot.imshow(img1, cmap=cmap)
    plot.subplot(1,2,2)
    plot.imshow(img2, cmap=cmap)
    plot.show()

#function to open and convert 2 images to grayscale
def open_image(path_to_image1, path_to_image2):
    img1 = cv2.imread(path_to_image1)
    img1_grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    img2 = cv2.imread(path_to_image2)
    img2_grey = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    return img1_grey, img2_grey

# open 2 images and convert them to grayscale

#for starters we begin with 2 random noise images just copy of eachother with a rotation
img1 = np.random.randint(0, 150, size=(150, 150, 3), dtype=np.uint8)
img2 = img1.copy()
img2 = rotate_image(img2, 1)
#plot the images
plot_images(img1, img2)

#convert the images to grayscale
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

plot_images(img1, img2, cmap='gray')

path_to_image1 = 'disparity/cporta_left.png'
path_to_image2 = 'disparity/cporta_right.png'

img1, img2 = open_image(path_to_image1, path_to_image2)
plot_images(img1, img2, cmap='gray')


#TODO : call lucaskanade function
#TODO : display the result
#TODO : save the result


