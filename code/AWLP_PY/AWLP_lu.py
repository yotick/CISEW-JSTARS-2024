import numpy as np
import cv2
from AWLP_PY.LPfilterGauss import LPfilterGauss
from AWLP_PY.estimation_alpha import estimation_alpha
from AWLP_PY.interp23tap import interp23tap


def MTF_GLP_AWLP_Haze(I_PAN, I_MS, ratio, decimation):
    vCorr = np.zeros(I_MS.shape[2])
    for ii in range(I_MS.shape[2]):
        b = I_MS[:, :, ii]
        vCorr[ii] = np.min(b)

    v3Corr = np.zeros((1, 1, I_MS.shape[2]))
    v3Corr[0, 0, :] = vCorr
    LCorr = np.tile(v3Corr, (I_MS.shape[0], I_MS.shape[1], 1))

    imageHR = I_PAN.astype(np.float)
    I_MS = I_MS.astype(np.float)

    ### Intensity
    imageHR_LP = LPfilterGauss(imageHR, ratio)

    h = estimation_alpha(I_MS, imageHR_LP, 'local')

    alpha = np.zeros((1, 1, I_MS.shape[2]))
    alpha[0, 0, :] = h[:, 0]  # 修改这一行
    I = np.sum((I_MS - LCorr) * np.tile(alpha, (I_MS.shape[0], I_MS.shape[1], 1)), axis=2)

    imageHR = (imageHR - np.mean(LPfilterGauss(imageHR, ratio))) * (
                np.std(I) / np.std(LPfilterGauss(imageHR, ratio))) + np.mean(I)

    # IntensityRatio = (I_MS - LCorr) / (np.tile(I, (1, 1, I_MS.shape[2])) + np.finfo(float).eps)
    IntensityRatio = (I_MS - LCorr) / (np.broadcast_to(np.expand_dims(I, axis=2), I_MS.shape) + np.finfo(float).eps)
    PAN_LP = LPfilterGauss(imageHR, ratio)
    if decimation:
        t = cv2.resize(PAN_LP, (round(PAN_LP.shape[1] / ratio), round(PAN_LP.shape[0] / ratio)),
                       interpolation=cv2.INTER_NEAREST)
        PAN_LP = interp23tap(t, ratio)

    # I_Fus_MTF_GLP_AWLP = I_MS + IntensityRatio * np.tile((imageHR - PAN_LP), (1, 1, I_MS.shape[2]))
    I_Fus_MTF_GLP_AWLP = I_MS + IntensityRatio * np.tile(np.expand_dims((imageHR - PAN_LP), axis=2),
                                                         (1, 1, I_MS.shape[2]))

    return I_Fus_MTF_GLP_AWLP
