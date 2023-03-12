import cv2
import matplotlib.pyplot as plot
import numpy as np
from ex1_utils import *
from lucas_Kanade import lucaskanade
from horn_Schunck import hornschuck

#function to open and convert 2 images to grayscale
def open_image(path_to_image1, path_to_image2):
    img1 = cv2.imread(path_to_image1)
    img1_grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    img2 = cv2.imread(path_to_image2)
    img2_grey = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    return img1_grey, img2_grey

# ========================= END OF FUNCTIONS =================================

# open 2 images and convert them to grayscale
path_to_image1 = 'lab2/001.jpg'
path_to_image2 = 'lab2/002.jpg'

img1_grey, img2_grey = open_image(path_to_image1, path_to_image2)

#plot the images
# plot_images(img1_grey, img2_grey, cmap='gray')


rnd_img1 = np.random.rand(200, 200).astype(np.float32)
rnd_img2 = rnd_img1.copy()
rnd_img2 = rotate_image(rnd_img2, -1)

# normalise the image
rnd_img1 = rnd_img1 / 255
rnd_img2 = rnd_img2 / 255
img1_grey = img1_grey / 255
img2_grey = img2_grey / 255

#call lucas-kanade
#u_lk , v_lk = lucaskanade(rnd_img1, rnd_img2, 10)
u_lk , v_lk = lucaskanade(img1_grey, img2_grey, 10)
print("konec lucas-kanade")

#display lucas-kanade flow and the images
#plot_flow(u_lk, v_lk, rnd_img1, rnd_img2, kateri='lk', save=True, filename='random-noise')

#TODO: Horn-Schunck

#call Horn-Schunck
#u_hs, v_hs = hornschuck(rnd_img1, rnd_img2, 1000, lmbd=0.5)
u_hs, v_hs = hornschuck(img1_grey, img2_grey, 1000, lmbd=0.5)
print("konec horn-schunck")

#dispaly Horn-Schunck flow and the images
#plot_flow(u_hs, v_hs, rnd_img1, rnd_img2, kateri='hs', save=True, filename='random-noise')

#cmb_plot_show(rnd_img1, img2_grey, u_lk, v_lk, u_hs, v_hs, 'random-noise')
cmb_plot_show(img1_grey, img2_grey, u_lk, v_lk, u_hs, v_hs, 'lab12')

#========== TESTING ===========