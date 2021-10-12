import random

import numpy as np
import skimage.color as sc
import torch
# from torch.utils.data import DataLoader
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
import logging
import os


###
###   Tools for argparse
###
# def str2list(s, split='_'):
#     l = s.split('_')
#     l = [int(x) for x in l]
#     return l

# def str2bool(s):
#     assert s in ['True', 'False']
#     if s == 'True':
#         return True
#     else:
#         return False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
###
### Tools for logging
###

def get_logger(logpath, displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info('\n\n------ ******* ------ New Log ------ ******* ------')
    return logger


###
### Tools for directories
###
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def scan_dir(dir, matching, fullPath=False):
    # Scan dir
    # Get all files matching particualr patterns: 8.png, [!._]*.png
    # if fullPath, return dir + matching; otherwise, return matching
    import glob
    file_list = glob.glob(os.path.join(dir, matching))
    if not fullPath:
        file_list = [os.path.split(x)[-1] for x in file_list]
    return file_list


###
### Tools for data processing
###
def get_patch(img_list, patch_size):
    # the input is a list of images
    # check dim, height, width
    # randomly select the x and y
    # the output is a list of patches
    dim = [img.ndim for img in img_list]
    assert(len(set(dim)) == 1)
    height = [img.shape[0] for img in img_list]
    assert(len(set(height)) == 1)
    width = [img.shape[1] for img in img_list]
    assert(len(set(width)) == 1)

    ih = random.randrange(0, height[0] - patch_size + 1)
    iw = random.randrange(0, width[0] - patch_size + 1)

    def _get_patch(img, patch_size, ih, iw):
        if img.ndim == 2:
            return img[ih:ih+patch_size, iw:iw+patch_size]
        else:
            return img[ih:ih+patch_size, iw:iw+patch_size,:]

    return [_get_patch(img, patch_size, ih, iw) for img in img_list]

def set_channel(l, n_channel):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        c = img.shape[2]
        if n_channel == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channel == 3 and c == 1:
            img = np.concatenate([img] * n_channel, 2)
        return img
    return [_set_channel(_l) for _l in l]

def np2Tensor(l, rgb_range):
    def _np2Tensor(img):
        if np.max(img) > 1:
            img = np.multiply(img, rgb_range / 255)

        if img.ndim == 2:
            img = set_channel([img], 1)[0]

        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        return tensor
    return [_np2Tensor(_l) for _l in l]

def augment(l):
    mode = np.random.randint(0, 8)
    # print(mode)
    def _augment(img, mode=0):
        if mode == 0:
            return img
        elif mode == 1:
            return np.flipud(img)
        elif mode == 2:
            return np.rot90(img)
        elif mode == 3:
            return np.flipud(np.rot90(img))
        elif mode == 4:
            return np.rot90(img, k=2)
        elif mode == 5:
            return np.flipud(np.rot90(img, k=2))
        elif mode == 6:
            return np.rot90(img, k=3)
        elif mode == 7:
            return np.flipud(np.rot90(img, k=3))

    return [_augment(_l, mode=mode) for _l in l]

def add_noise_numpy(x, param='.'):
    """ param should be [type, value] """
    if param is not '.':
        noise_type = param[0]
        noise_value = int(param[1])
        if noise_type == 'G':
            noises = np.random.normal(loc=0, scale=noise_value, size=x.shape)
            noises = noises.round()
        elif noise_type == 'S':
            noises = np.random.poisson(x * noise_value) / noise_value
            noises = noises - noises.mean(axis=0).mean(axis=0)
        x_noise = x.astype(np.int16) + noises.astype(np.int16)
        x_noise = x_noise.clip(0, 255).astype(np.uint8)
        return x_noise
    else:
        return x

def add_noise_tensor(x, param=['G',15]):
    """ param should be [type, value] """
    if param is not '.':
        noise_type = param[0]
        noise_value = int(param[1])

        if noise_type == 'G':
            noises = np.random.normal(loc=0, scale=noise_value, size=x.shape)
        elif noise_type == 'S':
            assert False, 'Please use Guassian Noises.'

        x_noise = x + torch.from_numpy(noises / 255).to(torch.float32)
        x_noise = torch.clamp(x_noise, 0, 1)
        return x_noise
    else:
        return x

