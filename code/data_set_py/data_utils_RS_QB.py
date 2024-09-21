from os import listdir
from os.path import join
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, Grayscale

# from general import histMatch
from data_set_py.imagecrop import FusionRandomCrop
from torchvision.transforms import functional as F
import numpy as np
from data_set_py.transforms import Stretch
from torch.nn import functional as FC
import scipy.io
from AWLP_PY.AWLP_lu import MTF_GLP_AWLP_Haze
from Pansharpening_Toolbox_Assessment_Python.MTF import lu_MTF


def data_transform():
    # return Compose([ToTensor(), Stretch()])
    return Compose([Stretch()])


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in
               ['.tif', '.bmp', '.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def tif_to_gray(image):
    # img = np.array(image)
    w, h, c = image.shape
    intensity = np.zeros((w, h))
    for i in range(c):
        intensity = intensity + 1 / c * image[:, :, i]

    # img = Image.fromarray(intensity)
    return intensity


def train_rand_crop(crop_size):
    return Compose([
        FusionRandomCrop(crop_size),
    ])


def train_gray_transform():
    return Compose([
        Grayscale(num_output_channels=3),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        #   Resize(400),
        #  CenterCrop(400),
        # Grayscale(),
        ToTensor()
    ])


def pil_crop_transform(pil, crop_size):
    img_crop = F.crop(pil, crop_size[0], crop_size[1], crop_size[2], crop_size[3])
    return ToTensor()(img_crop)


def detail_AWLP(I_PAN, I_MS):
    I_fused = MTF_GLP_AWLP_Haze(I_PAN, I_MS, 4, 0)
    detail_awlp = I_fused - I_MS
    return detail_awlp


class TestDatasetFromFolder(Dataset):  # for test datasets
    def __init__(self, dataset_dir, sate, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.gray_transform = train_gray_transform()

        self.sate = sate

        if sate == 'wv3_8':  # for test datasets
            # pan256_path = join(dataset_dir, 'WorldView3-8\\pan256\\')
            # pan1024_path = join(dataset_dir, 'WorldView3-8\\pan1024\\')
            # ms_r_path = join(dataset_dir, 'WorldView3-8\\mul64_MTF\\')
            # target_path = join(dataset_dir, 'WorldView3-8\\gt256\\')
            # ms_r_up_path = join(dataset_dir, 'WorldView3-8\\wv3_8_up\\')

            pan256_path = join(dataset_dir, 'data2017/DIV2K_valid_HR\\test_img7\PAN\PAN256')
            pan1024_path = join(dataset_dir, 'data2017/DIV2K_valid_HR\\test_img7\PAN\PAN1024')
            nir1_256_path = join(dataset_dir, 'data2017/DIV2K_valid_HR/test_img7/NIR1/NIR1256')
            nir2_256_path = join(dataset_dir, 'data2017/DIV2K_valid_HR/test_img7/NIR2/NIR2256')
            rgb256_path = join(dataset_dir, 'data2017/DIV2K_valid_HR/test_img7/RGB/RGB256')
            coastbl256_path = join(dataset_dir, 'data2017/DIV2K_valid_HR/test_img7/CoastalBlue/CoastalBlue256')
            rededge256_path = join(dataset_dir, 'data2017\DIV2K_valid_HR/test_img7\RedEdge\RedEdge256')
            yellow256_path = join(dataset_dir, 'data2017\DIV2K_valid_HR\\test_img7\Yellow\Yellow256')

            rgb64_path = join(dataset_dir, 'data2017/DIV2K_valid_LR_bicubic/X4/test_img7/RGB')
            nir1_64_path = join(dataset_dir, 'data2017/DIV2K_valid_LR_bicubic/X4/test_img7/NIR1')
            nir2_64_path = join(dataset_dir, 'data2017/DIV2K_valid_LR_bicubic/X4/test_img7/NIR2')
            coastbl64_path = join(dataset_dir, 'data2017/DIV2K_valid_LR_bicubic/X4/test_img7/CoastalBlue')
            rededge64_path = join(dataset_dir, 'data2017/DIV2K_valid_LR_bicubic/X4/test_img7/RedEdge')
            yellow64_path = join(dataset_dir, 'data2017/DIV2K_valid_LR_bicubic/X4/test_img7/yellow')

            # for test datasets
            self.nir1_64_file_name = [join(nir1_64_path, x.split('.')[0]) for x in listdir(nir1_64_path) if
                                      is_image_file(x)]
            self.nir2_64_file_name = [join(nir2_64_path, x.split('.')[0]) for x in listdir(nir2_64_path) if
                                      is_image_file(x)]
            self.coastbl64_file_name = [join(coastbl64_path, x.split('.')[0]) for x in listdir(coastbl64_path) if
                                        is_image_file(x)]
            self.yellow64_file_name = [join(yellow64_path, x.split('.')[0]) for x in listdir(yellow64_path) if
                                       is_image_file(x)]
            self.rededge64_file_name = [join(rededge64_path, x.split('.')[0]) for x in listdir(rededge64_path) if
                                        is_image_file(x)]
            self.nir1_256_file_name = [join(nir1_256_path, x.split('.')[0]) for x in listdir(nir1_256_path) if
                                       is_image_file(x)]
            self.nir2_256_file_name = [join(nir2_256_path, x.split('.')[0]) for x in listdir(nir2_256_path) if
                                       is_image_file(x)]
            self.coastbl256_file_name = [join(coastbl256_path, x.split('.')[0]) for x in listdir(coastbl256_path) if
                                         is_image_file(x)]
            self.yellow256_file_name = [join(yellow256_path, x.split('.')[0]) for x in listdir(yellow256_path) if
                                        is_image_file(x)]
            self.rededge256_file_name = [join(rededge256_path, x.split('.')[0]) for x in listdir(rededge256_path) if
                                         is_image_file(x)]

            # self.ms_r_file_name = join(ms_r_path, listdir(ms_r_path))
            # self.ms_r_file_name = [join(ms_r_path, x.split('.')[0]) for x in listdir(ms_r_path) if
            #                        is_image_file(x)]

        if sate == 'qb':  # for test datasets
            pan256_path = join(dataset_dir, 'QuickBird\\2_PAN256')
            nir256_path = join(dataset_dir, 'QuickBird\\2_NIR256')
            rgb256_path = join(dataset_dir, 'QuickBird\\2_RGB256')
            # pan1024_path = join(dataset_dir, 'Pleiades\\pl_pan1024\\')
            rgb64_path = join(dataset_dir, 'QuickBird\\2_RGB64_MTF')
            nir64_path = join(dataset_dir, 'QuickBird\\2_NIR64_MTF')

            self.nir64_file_name = [join(nir64_path, x.split('.')[0]) for x in listdir(nir64_path) if is_image_file(x)]
            self.nir256_file_name = [join(nir256_path, x.split('.')[0]) for x in listdir(nir256_path) if
                                     is_image_file(x)]
        if sate == 'ge':  # for test datasets
            pan256_path = join(dataset_dir, 'GeoEye_1\\PAN256')
            nir256_path = join(dataset_dir, 'GeoEye_1\\NIR256')
            rgb256_path = join(dataset_dir, 'GeoEye_1\\RGB256')
            # pan1024_path = join(dataset_dir, 'Pleiades\\pl_pan1024\\')
            rgb64_path = join(dataset_dir, 'GeoEye_1\\RGB64_MTF')
            nir64_path = join(dataset_dir, 'GeoEye_1\\NIR64_MTF')

            self.nir64_file_name = [join(nir64_path, x.split('.')[0]) for x in listdir(nir64_path) if is_image_file(x)]
            self.nir256_file_name = [join(nir256_path, x.split('.')[0]) for x in listdir(nir256_path) if
                                     is_image_file(x)]
            # target_path = join(dataset_dir, 'Pleiades\\pl_gt256\\')
            # ms_r_up_path = join(dataset_dir, 'Pleiades\\pl_up_ms\\')

        self.rgb64_file_name = [join(rgb64_path, x.split('.')[0]) for x in listdir(rgb64_path) if is_image_file(x)]
        self.pan256_file_name = [join(pan256_path, x.split('.')[0]) for x in listdir(pan256_path) if is_image_file(x)]
        self.rgb256_file_name = [join(rgb256_path, x.split('.')[0]) for x in listdir(rgb256_path) if is_image_file(x)]

        # self.target_file_name = [join(target_path, x.split('.')[0]) for x in listdir(target_path) if
        #                          is_image_file(x)]
        # self.ms_r_up_file_name = [join(ms_r_up_path, x.split('.')[0]) for x in listdir(ms_r_up_path) if
        #                           is_image_file(x)]

    def __getitem__(self, index):  # for test datasets
        pan256 = Image.open('%s.tif' % self.pan256_file_name[index])
        rgb256 = Image.open('%s.tif' % self.rgb256_file_name[index])
        rgb64 = Image.open('%s.tif' % self.rgb64_file_name[index])
        rgb_up = rgb64.resize((256, 256), Image.BICUBIC)
        rgb_up_near = rgb64.resize((256, 256), Image.NEAREST)

        rgb64_t = ToTensor()(rgb64)
        pan_crop = ToTensor()(pan256)
        rgb256_crop = ToTensor()(rgb256)
        rgb_up_crop = ToTensor()(rgb_up)
        rgb_near_crop = ToTensor()(rgb_up_near)

        if self.sate == 'qb' or self.sate == 'ge':  # for validation datasets
            nir256 = Image.open('%s.tif' % self.nir256_file_name[index])
            nir64 = Image.open('%s.tif' % self.nir64_file_name[index])
            nir_up = nir64.resize((256, 256), Image.BICUBIC)
            nir_up_near = nir64.resize((256, 256), Image.NEAREST)

            nir64_t = ToTensor()(nir64)
            nir256_crop = ToTensor()(nir256)
            nir_up_crop = ToTensor()(nir_up)
            nir_near_crop = ToTensor()(nir_up_near)

            gt_crop = torch.cat([rgb256_crop, nir256_crop])
            ms_up_crop = torch.cat([rgb_up_crop, nir_up_crop])
            ms_near_crop = torch.cat([rgb_near_crop, nir_near_crop])
            ms_64 = torch.cat([rgb64_t, nir64_t])

        if self.sate == 'wv3_8':  # for testing datasets
            nir1_64 = Image.open('%s.tif' % self.nir1_64_file_name[index])
            nir2_64 = Image.open('%s.tif' % self.nir2_64_file_name[index])
            nir1_256 = Image.open('%s.tif' % self.nir1_256_file_name[index])
            nir2_256 = Image.open('%s.tif' % self.nir2_256_file_name[index])
            coastbl64 = Image.open('%s.tif' % self.coastbl64_file_name[index])
            coastbl256 = Image.open('%s.tif' % self.coastbl256_file_name[index])
            rededge64 = Image.open('%s.tif' % self.rededge64_file_name[index])
            rededge256 = Image.open('%s.tif' % self.rededge256_file_name[index])
            yellow64 = Image.open('%s.tif' % self.yellow64_file_name[index])
            yellow256 = Image.open('%s.tif' % self.yellow256_file_name[index])

            nir1_up = nir1_64.resize((256, 256), Image.BICUBIC)
            coastbl_up = coastbl64.resize((256, 256), Image.BICUBIC)
            rededge_up = rededge64.resize((256, 256), Image.BICUBIC)
            yellow_up = yellow64.resize((256, 256), Image.BICUBIC)
            nir2_up = nir2_64.resize((256, 256), Image.BICUBIC)

            nir1_up_near = nir1_64.resize((256, 256), Image.NEAREST)  # for validation datasets
            coastbl_up_near = coastbl64.resize((256, 256), Image.NEAREST)
            rededge_up_near = rededge64.resize((256, 256), Image.NEAREST)
            yellow_up_near = yellow64.resize((256, 256), Image.NEAREST)
            nir2_up_near = nir2_64.resize((256, 256), Image.NEAREST)

            nir1_64_t = ToTensor()(nir1_64)
            nir1_256_crop = ToTensor()(nir1_256)
            nir1_up_crop = ToTensor()(nir1_up)
            nir1_near_crop = ToTensor()(nir1_up_near)

            coastbl64_t = ToTensor()(coastbl64)
            coastbl256_crop = ToTensor()(coastbl256)
            coastbl_up_crop = ToTensor()(coastbl_up)
            coastbl_near_crop = ToTensor()(coastbl_up_near)
            # for testing datasets
            rededge64_t = ToTensor()(rededge64)
            rededge256_crop = ToTensor()(rededge256)
            rededge_up_crop = ToTensor()(rededge_up)
            rededge_near_crop = ToTensor()(rededge_up_near)

            yellow64_t = ToTensor()((yellow64))
            yellow256_crop = ToTensor()(yellow256)
            yellow_up_crop = ToTensor()(yellow_up)
            yellow_near_crop = ToTensor()(yellow_up_near)

            nir2_64_t = ToTensor()(nir2_64)
            nir2_256_crop = ToTensor()(nir2_256)
            nir2_up_crop = ToTensor()(nir2_up)
            nir2_near_crop = ToTensor()(nir2_up_near)

            gt_crop = torch.cat(
                [rgb256_crop, nir1_256_crop, coastbl256_crop, rededge256_crop, yellow256_crop, nir2_256_crop])
            ms_up_crop = torch.cat(
                [rgb_up_crop, nir1_up_crop, coastbl_up_crop, rededge_up_crop, yellow_up_crop, nir2_up_crop])
            ms_near_crop = torch.cat(
                [rgb_near_crop, nir1_near_crop, coastbl_near_crop, rededge_near_crop, yellow_near_crop,
                 nir2_near_crop])  # for testing datasets
            ms_64 = torch.cat([rgb64_t, nir1_64_t, coastbl64_t, rededge64_t, yellow64_t, nir2_64_t])
        ms_org_crop = ms_64

        # data = torch.cat([ms_gray_crop, pan_crop])
        # return ms_up_crop, detail_crop, detail_gt_crop
        return ms_up_crop, ms_org_crop, pan_crop, gt_crop

        # data = torch.cat([ms_gray_crop, pan_crop])
        # return ms_up_crop, detail_crop, detail_gt_crop
        # return ms_up_crop, ms_org_crop, pan_crop, gt_crop

    def __len__(self):
        return len(self.rgb256_file_name)
