import numpy as np

def estimation_alpha(I_MS, I_PAN, type_estimation):
    if type_estimation == 'global':
        # Global estimation
        IHc = I_PAN.reshape(-1, 1)
        ILRc = I_MS.reshape(-1, I_MS.shape[2])
        alpha = np.linalg.lstsq(ILRc, IHc, rcond=None)[0]
    else:
        # Local estimation
        block_win = 32
        alphas = np.zeros((I_MS.shape[2], 1))
        cont_bl = 0
        for ii in range(0, I_MS.shape[0], block_win):
            for jj in range(0, I_MS.shape[1], block_win):
                imHRbl = I_PAN[ii:min(I_MS.shape[0], ii + block_win),
                                 jj:min(I_MS.shape[1], jj + block_win)]
                imageLRbl = I_MS[ii:min(I_MS.shape[0], ii + block_win),
                                  jj:min(I_MS.shape[1], jj + block_win), :]
                imageHRc = imHRbl.reshape(-1, 1)
                ILRc = imageLRbl.reshape(-1, imageLRbl.shape[2])
                alphah = np.linalg.lstsq(ILRc, imageHRc, rcond=None)[0]
                alphas += alphah
                cont_bl += 1
        alpha = alphas / cont_bl
    return alpha