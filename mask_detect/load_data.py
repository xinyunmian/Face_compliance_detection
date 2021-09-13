import numpy as np
import torch
import torch.utils.data as data
import cv2
from myconfig import config
import random

def mirror_face(img):
    mir_img = img[:, ::-1]
    return mir_img

def pinghua_face(img):
    blur_img = cv2.GaussianBlur(img, (5,5), 0)
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

def resize_crop_face(img):
    imgh, imgw, imgc = img.shape
    scale = random.uniform(0.9, 1.0)
    new_w = int(scale * imgw)
    new_h = int(scale * imgh)
    if new_w == imgw:
        l = 0
    else:
        l = random.randrange(imgw - new_w)
    if new_h == imgh:
        t = 0
    else:
        t = random.randrange(imgh - new_h)
    roi = np.array((l, t, l + new_w, t + new_h))
    img_crop = img[roi[1]:roi[3], roi[0]:roi[2]]
    return img_crop

class Data_augment(object):
    def __init__(self, aug, mir, ph, zq, rot, crop):
        self.aug = aug
        self.mir = mir
        self.ph = ph
        self.zq = zq
        self.rot = rot
        self.crop = crop
    def __call__(self, image):
        if self.aug > 0:
            if self.mir > 0:
                image = mirror_face(image)
            if self.ph > 0:
                image = pinghua_face(image)
            if self.zq > 0:
                a = random.choice([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
                b = random.choice([5, 8, 10, 11, 12, 15, 18, 20, 25, 30])
                image = zengqiang_face(image, a, b)
            if self.rot > 0:
                angle = random.choice([-10, -5, 0, 5, 10])
                image = rotate_face(image, angle)
            if self.crop > 0:
                image = resize_crop_face(image)
        return image

class Face_classification(data.Dataset):
    def __init__(self, dir_path, txt_path):
        with open(txt_path, 'r') as f:
            self.imgs = list(map(lambda line: line.strip().split(' '), f))
        self.dir_path = dir_path

    def __getitem__(self, index):
        path, label = self.imgs[index]
        # print(path,label)
        img_path = self.dir_path + "/" + path
        img = cv2.imread(img_path)
        augment_ornot = random.choice([0, 1])
        mirror_ornot = random.choice([0, 1, 2])
        blur_ornot = random.choice([0, 1, 2])
        light_ornot = random.choice([0, 1, 2])
        rotate_ornot = random.choice([0, 1, 2])
        crop_ornot = random.choice([0, -1, -2])
        process = Data_augment(augment_ornot, mirror_ornot, blur_ornot, light_ornot, rotate_ornot, crop_ornot)
        img = process(img)

        img = cv2.resize(img, (config.img_height, config.img_width), interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32)
        img = (img - config.bgr_mean) / config.bgr_std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        label = int(label)
        return img, label

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)
