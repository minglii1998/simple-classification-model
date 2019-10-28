import os, sys, shutil
import random as rd
from PIL import Image, ImageEnhance
import numpy as np
import pdb
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss


def randomRotation(image, mode=Image.BICUBIC):
    random_angle = np.random.randint(1, 360)
    return image.rotate(random_angle, mode)


def randomFlip(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def randomColor(image):
    random_factor = np.random.randint(10, 21) / 10.
    brightness_image = ImageEnhance.Brightness(image).enhance(random_factor)
    random_factor = np.random.randint(10, 21) / 10.
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(
        random_factor)
    randon_factor = np.random.randint(0, 31) / 10.
    sharpness_image = ImageEnhance.Sharpness(contrast_image).enhance(
        random_factor)
    return sharpness_image


def randomGaussian(image, mean=0.2, sigma=0.3):
    def GaussianNoise(im, mean=0.2, sigma=0.3):
        for _i in range(len(im)):
            im[_i] += rd.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    # print(img.flags)
    img = np.require(img, requirements=['C', 'A', 'W'])
    img.flags.writeable = True
    width, height = img.shape[:2]
    img_r = GaussianNoise(img[:, :, 0].flatten(), mean, sigma)
    img_g = GaussianNoise(img[:, :, 1].flatten(), mean, sigma)
    img_b = GaussianNoise(img[:, :, 2].flatten(), mean, sigma)
    img[:, :, 0] = img_r.reshape([width, height])
    img[:, :, 1] = img_g.reshape([width, height])
    img[:, :, 2] = img_b.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def data_aug(img):

    final_size = 300
    final_width = final_height = final_size
    crop_size = 200
    crop_height = crop_width = crop_size
    crop_center_y_offset = 15
    crop_center_x_offset = 0

    scale_aug = 0.02
    trans_aug = 0.01
    p = []
    for i in range(5):
        p.append(rd.randint(0, 2))

    if p[0] == 1:
        img = randomRotation(img)
    if p[1] == 1:
        img = randomFlip(img)
    if p[2] == 1:
        img = randomColor(img)
    if p[3] == 1:
        img = randomGaussian(img)

    # computed parameters
    randint = rd.randint
    scale_height_diff = (randint(0, 1000) / 500 - 1) * scale_aug
    crop_height_aug = crop_height * (1 + scale_height_diff)
    scale_width_diff = (randint(0, 1000) / 500 - 1) * scale_aug
    crop_width_aug = crop_width * (1 + scale_width_diff)

    trans_diff_x = (randint(0, 1000) / 500 - 1) * trans_aug
    trans_diff_y = (randint(0, 1000) / 500 - 1) * trans_aug

    center = ((img.width / 2 + crop_center_x_offset) * (1 + trans_diff_x),
              (img.height / 2 + crop_center_y_offset) * (1 + trans_diff_y))

    if center[0] < crop_width_aug / 2:
        crop_width_aug = center[0] * 2 - 0.5
    if center[1] < crop_height_aug / 2:
        crop_height_aug = center[1] * 2 - 0.5
    if (center[0] + crop_width_aug / 2) >= img.width:
        crop_width_aug = (img.width - center[0]) * 2 - 0.5
    if (center[1] + crop_height_aug / 2) >= img.height:
        crop_height_aug = (img.height - center[1]) * 2 - 0.5

    crop_box = (center[0] - crop_width_aug / 2,
                center[1] - crop_height_aug / 2,
                center[0] + crop_width_aug / 2, center[1] + crop_width_aug / 2)

    mid_img = img.crop(crop_box)
    image = img.resize((final_width, final_height))

    return image

if __name__ == '__main__':
    img = Image.open('1.png')
    img2 = data_aug(img)
    img2.show()