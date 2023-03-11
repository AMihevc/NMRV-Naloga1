from ex1_utils import *
import numpy as np
from scipy.signal import convolve2d 

def hornschuck(img1, img2, n_iters, lmbd ):
    #im1 - first image matrix (grayscale)
    #im2 - second image matrix (grayscale)
    #iter - number of iterations 
    #lamb - paramater lambda

    ix, iy, it = calc_derivatives(img1, img2)
    


    return 0 