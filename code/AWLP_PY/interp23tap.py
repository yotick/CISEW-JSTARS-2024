import numpy as np
from scipy import signal


def interp23tap(I_Interpolated, ratio):
    if 2 ** round(np.log2(ratio)) != ratio:
        print('Error: Only resize factors power of 2')
        return

    r, c, b = I_Interpolated.shape
    CDF23 = 2.0 * np.array(
        [0.5, 0.305334091185, 0, -0.072698593239, 0, 0.021809577942, 0, -0.005192756653, 0, 0.000807762146, 0,
         -0.000060081482])
    CDF23 = np.concatenate((np.flip(CDF23[1:]), CDF23))
    BaseCoeff = CDF23
    first = True

    for z in range(1, int(ratio / 2) + 1):
        I1LRU = np.zeros(((2 ** z) * r, (2 ** z) * c, b))

        if first:
            I1LRU[1::2, 1::2, :] = I_Interpolated
            first = False
        else:
            I1LRU[::2, ::2, :] = I_Interpolated

        for ii in range(b):
            t = I1LRU[:, :, ii]
            t = signal.convolve(t.T, BaseCoeff, mode='same', method='direct')
            I1LRU[:, :, ii] = signal.convolve(t.T, BaseCoeff, mode='same', method='direct')

        I_Interpolated = I1LRU

    return I_Interpolated