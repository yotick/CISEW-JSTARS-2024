import os
import glob
import numpy as np
import scipy.io as sio
from torch import nn
from torch.nn import functional as FC
from torchvision.transforms import ToTensor
from Pansharpening_Toolbox_Assessment_Python.imresize import imresize
from Pansharpening_Toolbox_Assessment_Python.MTF import MTF

import torch
# (sensor == 'QB') or (sensor == 'IKONOS') or (sensor == 'GeoEye1') or (sensor == 'WV2') or (sensor == 'WV3') or (
# sensor == 'WV4'):
sensor = 'WV3'
# 指定待读取和保存的文件夹路径
""" Resize Factor """
ratio = 4

data_dir = ''
save_dir = ''
if sensor == 'WV3':
    data_dir = r'F:\remote sense image fusion\Source Images\data2017\DIV2K_valid_HR\test_img7\gt256'
    save_dir = r'F:\remote sense image fusion\Source Images\data2017\DIV2K_valid_LR_bicubic\X4\test_img7\mul64_MTF_conv'

# 使用glob获取文件夹下所有的.mat文件
files = glob.glob(os.path.join(data_dir, '*.mat'))

# 遍历每个.mat文件
for file in files:
    # 读取.mat文件内容
    ms = sio.loadmat(file).get('gt')
    ms = ToTensor()(ms).cuda()
    # 对.mat文件内容做一些修改
    ################## the first method:
    # I_MS_LP = MTF(ms, sensor, ratio)
    #
    # """ Decimation MS"""
    # I_MS_LP_D = I_MS_LP[int(ratio / 2):-1:int(ratio), int(ratio / 2):-1:int(ratio), :].astype(np.uint8)

    ###################### the second method:
    # MTF and downsample first, then upsample.
    c, h, w = ms.shape
    ms_kernel_name = './kernels/WV3_ms_kernel.mat'  # read the corresponding multispectral kernel (WorldView-3
    # (8x8*7x7), QuickBird and GaoFen-2 (4x4x7x7))
    ms_kernel = sio.loadmat(ms_kernel_name)
    ms_kernel = ms_kernel['ms_kernel'][...]  # get key and value
    ms_kernel = torch.FloatTensor(ms_kernel).cuda()  ## change to float tensor type
    # ms_kernel = torch.FloatTensor(ms_kernel).unsqueeze(0).cuda()  ## 1 7 7 4 change to float tensor type
    # ms_kernel = ms_kernel.permute(3, 0, 1, 2)  # 4 1 7  7
    # ms_kernel = ms_kernel.permute(2,0,1)

    # ms_kernel_name = './kernels/none_ms_kernel.mat'  # read the corresponding multispectral kernel (WorldView-3
    # # (8x8*7x7), QuickBird and GaoFen-2 (4x4x7x7))
    # ms_kernel = sio.loadmat(ms_kernel_name)
    # ms_kernel = ms_kernel['ms_kernel'][...]  # get key and value
    # ms_kernel = torch.FloatTensor(ms_kernel).cuda()   ## change to float tensor type

    mtf_filter = nn.Conv2d(in_channels=c, out_channels=c,
                           kernel_size=7, groups=1, bias=False, padding=3, padding_mode='replicate')

    mtf_filter.weight.data = ms_kernel
    mtf_filter.weight.requires_grad = False
    ms_fd = mtf_filter(ms).unsqueeze(0)
    ms_mtf_d = FC.interpolate(ms_fd, size=(h // 4, w // 4), mode='nearest')  ## change here!! nearest

    # ms_d_up = FC.interpolate(ms_mtf_d, size=(h, w), mode='bicubic')

    # 将张量转换为NumPy数组，并将其类型设置为np.uint8
    array = (ms_mtf_d.squeeze(0).permute(1,2,0).cpu() * 255)
    array = array.to(torch.uint8).numpy()

    # 将修改后的内容保存到指定文件夹中
    # 获取文件名
    file_name = os.path.basename(file)
    # 拼接保存路径
    save_path = os.path.join(save_dir, file_name)
    # 保存为.mat文件
    sio.savemat(save_path, {'mul64':array})
