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
rnd_img2 = rotate_image(rnd_img2, -1)

# normalise 
rnd_img1 = rnd_img1 / 255
rnd_img2 = rnd_img2 / 255


u_lk , v_lk = lucaskanade(rnd_img1, rnd_img2, 3)

fig1, ((ax_11, ax_12), (ax_21, ax_22)) = plot.subplots(2, 2)

ax_11.imshow(rnd_img1)
ax_11.set_title("Prva slika")
ax_12.imshow(rnd_img2)
ax_12.set_title("Naslednja slika")

show_flow(u_lk, v_lk, ax_21)
plot.show()

#TODO : display the result
#TODO : save the result


#========== TESTING ===========

# ix, iy, it = calc_derivatives(img1_grey, img2_grey)

# plot_images(ix, iy, cmap='gray')

#plot just 1 image for testing
# plot.imshow(it, cmap='gray')
# plot.show()