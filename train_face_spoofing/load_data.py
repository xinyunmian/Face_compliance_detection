import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.utils.data.dataloader as DataLoader
import torchvision.transforms as transforms
import cv2
from myconfig import config
import random

def mirror_face(img):
    mir_img = img[:, ::-1]
    return mir_img

def pinghua_face(img):
    blur_img = cv2.gaussianBlur(img, 3)
    return blur_img

def zengqiang_face(img, a, b):
    res = np.uint8(np.clip((a * img + b), 0, 255))
    return res

def rotate_face(img, angle):
    imgh, imgw, imgc = img.shape
    center = (imgw / 2, imgh / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (imgw, imgh))
    return rotated


class Face_classification(data.Dataset):
    def __init__(self, dir_path, txt_path):
        with open(txt_path, 'r') as f:
            self.imgs = list(map(lambda line: line.strip().split(' '), f))
        self.dir_path = dir_path

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img_path = self.dir_path + "/" + path
        img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#HWC
        # img = torch.from_numpy(img)#tensor
        img = cv2.resize(img, (config.img_height, config.img_width), interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32)
        img -= config.rgb_mean
        # img = np.transpose(img, (2, 0, 1))  # CHW
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        label = int(label)
        return img, label

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)
