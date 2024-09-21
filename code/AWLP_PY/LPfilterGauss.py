from scipy import signal
from skimage import filters
import numpy as np

def LPfilterGauss(I_PAN, ratio):
    GNyq = 0.3
    N = 41
    fcut = 1 / ratio

    alpha = np.sqrt((N * (fcut / 2)) ** 2 / (-2 * np.log(GNyq)))
    H = filters.gaussian(np.array([N, N]), alpha)
    Hd = H / np.max(H)
    h = signal.firwin(N, cutoff=fcut / 2, window=('kaiser', 0.5))
    h_2d = np.reshape(h, (1, N))
    I_PAN_LR = signal.convolve2d(I_PAN, h_2d, mode='same', boundary='symm')
    return I_PAN_LR