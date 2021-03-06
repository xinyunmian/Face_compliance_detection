import numpy as np
import torch
import torch.utils.data as data
import cv2
from myconfig import config
import random
import math

def normal_sampling(mean, label_k, std=2):
    return math.exp(-(label_k-mean)**2/(2*std**2))/(math.sqrt(2*math.pi)*std)

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
    scale = random.uniform(0.85, 1.0)
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

class Age_GenderDataLoader(data.Dataset):
    def __init__(self, dir_path, txt_path):
        with open(txt_path, 'r') as f:
            self.imgs = list(map(lambda line: line.strip().split(' '), f))
        self.dir_path = dir_path

    def __getitem__(self, index):
        if (len(self.imgs[index]) > 3 or len(self.imgs[index]) < 3):
            print(self.imgs[index])
        path, agelab, genderlab = self.imgs[index]
        if (int(genderlab) < 0 or int(genderlab) > 1):
            print(self.imgs[index])
        img_path = self.dir_path + "/" + path
        img = cv2.imread(img_path)
        if img is None:
            print(self.imgs[index])
        augment_ornot = random.choice([0, 1])
        mirror_ornot = random.choice([0, 1, 2])
        blur_ornot = random.choice([0, 1, 2])
        light_ornot = random.choice([0, 1, 2])
        rotate_ornot = random.choice([0, 1, 2])
        crop_ornot = random.choice([0, 1, 2])
        process = Data_augment(augment_ornot, mirror_ornot, blur_ornot, light_ornot, rotate_ornot, crop_ornot)
        img = process(img)

        img = cv2.resize(img, (config.img_height, config.img_width), interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32)
        img = (img - config.bgr_mean) / config.bgr_std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        agelab = float(agelab) / 116.0
        agelab = float(agelab)
        genderlab = int(genderlab)
        return img, agelab, genderlab

    def __len__(self):  # ???????????????????????????????????????????????????????????????????????????????????????????????????loader??????????????????
        return len(self.imgs)

class AG_DLDLDataLoader(data.Dataset):
    def __init__(self, dir_path, txt_path):
        with open(txt_path, 'r') as f:
            self.imgs = list(map(lambda line: line.strip().split(' '), f))
        self.dir_path = dir_path

    def __getitem__(self, index):
        if (len(self.imgs[index]) > 3 or len(self.imgs[index]) < 3):
            print(self.imgs[index])
        path, agelab, genderlab = self.imgs[index]
        if (int(genderlab) < 0 or int(genderlab) > 1):
            print(self.imgs[index])
        img_path = self.dir_path + "/" + path
        img = cv2.imread(img_path)
        if img is None:
            print(self.imgs[index])
        augment_ornot = random.choice([0, 1])
        mirror_ornot = random.choice([0, 1, 2])
        blur_ornot = random.choice([0, 1, 2])
        light_ornot = random.choice([0, 1, 2])
        rotate_ornot = random.choice([0, 1, 2])
        crop_ornot = random.choice([0, 1, 2])
        process = Data_augment(augment_ornot, mirror_ornot, blur_ornot, light_ornot, rotate_ornot, crop_ornot)
        img = process(img)

        img = cv2.resize(img, (config.img_height, config.img_width), interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32)
        img = (img - config.bgr_mean) / config.bgr_std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        agelab = int(agelab)
        normlabel = [normal_sampling(agelab, i) for i in range(116)]
        normlabel = [i if i > 1e-15 else 1e-15 for i in normlabel]
        normlabel = torch.Tensor(normlabel)

        genderlab = int(genderlab)
        return img, agelab, normlabel, genderlab

    def __len__(self):  # ???????????????????????????????????????????????????????????????????????????????????????????????????loader??????????????????
        return len(self.imgs)
