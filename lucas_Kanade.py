from ex1_utils import *
import numpy as np
from scipy.signal import convolve2d 



def lucaskanade ( im1 , im2 , N=3 ): 
    #im1 - first image matrix (grayscale)
    #im2 - second image matrix (grayscale)
    #N - size of the neighborhood (NxN) default is 3 so a 3x3 neighborhood

    #calculate the dirivaties Ix Iy and It
    ix, iy, it = calc_derivatives(im1, im2)

    # components of u, v and D 
    # convolution with kernal of size NxN 
    jedroSosedov = np.ones((N, N), dtype=np.float32) / (N * N)

    ix_x = convolve2d(np.multiply(ix, ix), jedroSosedov, mode='same')
    iy_y = convolve2d(np.multiply(iy, iy), jedroSosedov, mode='same')
    
    ix_t = convolve2d(np.multiply(ix, it), jedroSosedov, mode='same')
    iy_t = convolve2d(np.multiply(iy, it), jedroSosedov, mode='same')

    ix_y = convolve2d(np.multiply(ix, iy), jedroSosedov, mode='same')

    # determinant of a covariance matrix 
    d = np.subtract( np.multiply(ix_x, iy_y) , np.multiply(ix_y, ix_y))

    #some elements are 0 which meses up the division 
    # add some small number to all elements so there is no division by zero
    d += 1e-5 
    
    #zaznamek nism zih za te formule morta met še minus uspredi 
    u = np.divide(
            np.subtract(
                np.multiply(iy_y, ix_t), 
                np.multiply(ix_y, iy_t)
            ),
        d)
    
    v = np.divide(
        np.subtract(
            np.multiply(ix_x, iy_t), 
            np.multiply(ix_y, ix_t)
        ),
    d)
    
    return u, v 

