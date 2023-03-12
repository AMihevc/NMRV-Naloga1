from ex1_utils import *
import numpy as np
from scipy.signal import convolve2d 
from sklearn.metrics.pairwise import cosine_similarity

def hornschuck(img1, img2, n_iters, lmbd ):
    #im1 - first image matrix (grayscale)
    #im2 - second image matrix (grayscale)
    #iter - number of iterations 
    #lamb - paramater lambda

    #zracunaj odvode
    ix, iy, it = calc_derivatives(img1, img2)

    #from instructions: The initial estimates for u and v are typically set to 0 and are then iteratively improved
    u, v = np.zeros(img1.shape), np.zeros(img2.shape)

    # print("ix shape:")
    # print(ix.shape)
    # print("img shape:")
    # print(img1.shape)

    #potrebni elementi za enacbe 
    u_a = np.ones(img1.shape)
    v_a = np.ones(img1.shape)

    #residual Laplacian kernel
    l_d = np.matrix([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])

    #the D matrix 
    d = np.square(ix) + np.square(iy) + lmbd

    #iteration loop
    debug = 0
    for i in range(n_iters):

        #convolution
        u_a = convolve2d(u ,l_d, mode='same')
        v_a = convolve2d(v ,l_d, mode='same')

        #caluclate P 
        p = np.add (it , np.multiply(ix, u_a) , np.multiply(iy, v_a))
        p_div_d = np.divide(p, d)

        u = np.subtract(u_a, np.multiply(ix, p_div_d))
        v = np.subtract(v_a, np.multiply(iy, p_div_d))

    
    #print("Horn-Schunck iterations: " + str(debug))
    return u, v