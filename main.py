import cv2
import matplotlib.pyplot as plot
import numpy as np
from ex1_utils import *
from lucas_Kanade import lucaskanade


#function to open and convert 2 images to grayscale
def open_image(path_to_image1, path_to_image2):
    img1 = cv2.imread(path_to_image1)
    img1_grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    img2 = cv2.imread(path_to_image2)
    img2_grey = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    return img1_grey, img2_grey

# ========================= END OF FUNCTIONS =================================

# open 2 images and convert them to grayscale
# path_to_image1 = 'lab2/001.jpg'
# path_to_image2 = 'lab2/002.jpg'

# img1_grey, img2_grey = open_image(path_to_image1, path_to_image2)

# #plot the images
# plot_images(img1_grey, img2_grey, cmap='gray')


#for testing we begin with 2 random noise images just copy of eachother with a rotation
# rnd_img1 = np.random.randint(0, 150, size=(150, 150, 3), dtype=np.uint8)
# rnd_img2 = rnd_img1.copy()
# rnd_img2 = rotate_image(rnd_img2, 1)

# #convert the images to grayscale
# img1_grey = cv2.cvtColor(rnd_img1, cv2.COLOR_BGR2GRAY)
# img2_grey = cv2.cvtColor(rnd_img2, cv2.COLOR_BGR2GRAY)

# #plot the images
# plot_images(rnd_img1, rnd_img2, cmap='gray')

rnd_img1 = np.random.rand(200, 200).astype(np.float32)
rnd_img2 = rnd_img1.copy()
rnd_img2 = rotate_image(rnd_img2, 1)

# normalise the image
rnd_img1 = rnd_img1 / 255
rnd_img2 = rnd_img2 / 255

#call lucas-kanade
u_lk , v_lk = lucaskanade(rnd_img1, rnd_img2, 3)

#display lucas-kanade flow and the images
plot_flow(u_lk, v_lk, rnd_img1, rnd_img2)

#TODO: Horn-Schunck

#TODO call Horn-Schunck

#TODO dispaly Horn-Schunck flow and the images

#TODO save the plots of both images
#to lahko mogoče še ko displayaš najprej lk potem hs

#========== TESTING ===========