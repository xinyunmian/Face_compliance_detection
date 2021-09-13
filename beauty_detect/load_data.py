import numpy as np
import torch
import torch.utils.data as data
import cv2
from myconfig import config
import random
import math

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

class LoadDataBeauty(data.Dataset):
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
        mirror_ornot = random.choice([0, 1, 2, 3])
        blur_ornot = random.choice([0, 1, 2, 3])
        light_ornot = random.choice([0, 1])
        rotate_ornot = random.choice([0, 1])
        crop_ornot = random.choice([0, 1])
        process = Data_augment(augment_ornot, mirror_ornot, 0, 0, rotate_ornot, 0)
        img = process(img)

        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = interp_methods[random.randrange(5)]
        img = cv2.resize(img, (config.img_width, config.img_height), interpolation=interp_method)
        img = img.astype(np.float32)
        img = img / 255.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        beauty = int(label)
        lab = self.label_change(beauty)
        return img, lab

    def label_change(self, lab):
        change_lab = 0.0
        if lab == 0:
            change_lab = 0.0
        if lab == 1:
            change_lab = 0.2
        if lab == 2:
            change_lab = 0.4
        if lab == 3:
            change_lab = 0.6
        if lab == 4:
            change_lab = 0.8
        if lab == 5:
            change_lab = 1.0
        return change_lab

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)

def Beauty_collate(batch):
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            else:
                annos = torch.tensor(tup).float()
                targets.append(annos)
    return (torch.stack(imgs, 0), torch.stack(targets, 0))

class pytorch_to_dpcoreParams():
    def __init__(self, net):
        super(pytorch_to_dpcoreParams, self).__init__()
        self.net = net

    def save_param_name_pytorch(self, file, name, dim, model_names):
        if name.endswith("weight"):
            splitn = name.split(".weight")[0]
        if name.endswith("bias"):
            splitn = name.split(".bias")[0]

        cfg_begin = "const float* const PLH_" + splitn.upper().replace(".", "_") + "[] = " + "\n" + "{" + "\n"
        if name.endswith("weight"):
            file.write(cfg_begin)

        kg4 = "    "
        namesplit = splitn.upper().replace(".", "_")
        if dim == 2 and name.endswith("weight"):
            biasname = splitn + ".bias"
            str_name1 = namesplit + "_WEIGHT," + "\n"
            file.write(kg4)
            file.write(str_name1)
            if biasname in model_names:
                file.write(kg4)
                str_name2 = namesplit + "_BIAS," + "\n"
                file.write(str_name2)
            endstr = "}" + ";" + "\n" + "\n"
            file.write(endstr)
        if dim == 4 and name.endswith("weight"):
            biasname = splitn + ".bias"
            str_name1 = namesplit + "_WEIGHT," + "\n"
            file.write(kg4)
            file.write(str_name1)
            if biasname in model_names:
                file.write(kg4)
                str_name2 = namesplit + "_BIAS," + "\n"
                file.write(str_name2)
            else:
                file.write(kg4)
                file.write("NULL," + "\n")
            endstr = "}" + ";" + "\n" + "\n"
            file.write(endstr)
        if dim == 1 and name.endswith("weight"):
            str_name1 = namesplit + "_WEIGHT," + "\n"
            str_name2 = namesplit + "_BIAS," + "\n"
            file.write(kg4)
            file.write(str_name1)
            file.write(kg4)
            file.write(str_name2)
            file.write(kg4)
            file.write("NULL," + "\n")
            file.write(kg4)
            file.write("NULL," + "\n")
            endstr = "}" + ";" + "\n" + "\n"
            file.write(endstr)

    def save_convparam_(self, file, splitname, model_names):
        weightname = splitname + ".weight"
        biasname = splitname + ".bias"
        weight = self.net.state_dict()[weightname]

        w_name = splitname.upper().replace(".", "_") + "_WEIGHT" # CONV1_WEIGHT
        src_begin = "const float " + w_name + "[] = " + "\n" + "{" + "\n"
        file.write(src_begin)

        feadata = weight.data.cpu().numpy()
        ochannel, ichannel, height, width = feadata.shape
        kg4 = " "
        dimstr = "//    " + str(ochannel) + ", " + str(ichannel) + ", " + str(height) + ", " + str(width) + "\n"
        file.write(dimstr)
        for l in range(ochannel):
            ffdata = feadata[l, :, :, :]
            for i in range(ichannel):
                for j in range(height):
                    for k in range(width):
                        fdata = ffdata[i, j, k]
                        file.write(kg4)
                        if fdata >= 0:
                            sdata = ('%.6f' % fdata)
                            file.write("+" + sdata + "f,")
                        if fdata < 0:
                            sdata = ('%.6f' % fdata)
                            file.write(sdata + "f,")
                file.write("\n")
            file.write("\n")
        endstr = "}" + ";" + "\n" + "\n"
        file.write(endstr)

        if biasname in model_names:
            b_name = splitname.upper().replace(".", "_") + "_BIAS"  # CONV_BIAS
            b_begin = "const float " + b_name + "[] = " + "\n" + "{" + "\n"
            file.write(b_begin)

            bias = self.net.state_dict()[biasname]
            biasdata = bias.data.cpu().numpy()
            dimstr1 = "//    " + str(ochannel) + "," + "\n"
            file.write(dimstr1)
            #####save bias
            for v in range(int(ochannel)):
                bdata = biasdata[v]
                file.write(kg4)
                if bdata >= 0:
                    sdata = ('%.6f' % bdata)
                    file.write("+" + sdata + "f,")
                if bdata < 0:
                    sdata = ('%.6f' % bdata)
                    file.write(sdata + "f,")
            file.write("\n")
            endstr = "}" + ";" + "\n" + "\n"
            file.write(endstr)

    def save_bnparam(self, file, splitname, weight, bias, mean, var, eps=1e-5):
        w_name = splitname.upper().replace(".", "_") + "_WEIGHT"  # CONV1_WEIGHT
        w_begin = "const float " + w_name + "[] = " + "\n" + "{" + "\n"
        b_name = splitname.upper().replace(".", "_") + "_BIAS"  # CONV1_WEIGHT
        b_begin = "const float " + b_name + "[] = " + "\n" + "{" + "\n"
        file.write(w_begin)

        weidata = weight.data.cpu().numpy()
        biasdata = bias.data.cpu().numpy()
        meandata = mean.data.cpu().numpy()
        vardata = var.data.cpu().numpy()
        ochannel = weidata.shape
        kg4 = " "
        dimstr = "//    " + str(ochannel) + "," + "\n"
        file.write(dimstr)
        ochannel = ochannel[0]

        #####save weight
        for l in range(int(ochannel)):
            wdata = weidata[l]
            vdata = vardata[l]
            file.write(kg4)
            wdata = wdata / (math.sqrt(vdata) + eps)
            if wdata >= 0:
                sdata = ('%.6f' % wdata)
                file.write("+" + sdata + "f,")
            if wdata < 0:
                sdata = ('%.6f' % wdata)
                file.write(sdata + "f,")
        endstr = "\n" + "}" + ";" + "\n" + "\n"
        file.write(endstr)

        file.write(b_begin)
        dimstr1 = "//    " + str(ochannel) + "," + "\n"
        file.write(dimstr1)
        #####save bias
        for v in range(int(ochannel)):
            wdata = weidata[v]
            vdata = vardata[v]
            bdata = biasdata[v]
            mdata = meandata[v]
            file.write(kg4)
            bdata = bdata - (wdata * mdata) / (math.sqrt(vdata) + eps)
            if bdata >= 0:
                sdata = ('%.6f' % bdata)
                file.write("+" + sdata + "f,")
            if bdata < 0:
                sdata = ('%.6f' % bdata)
                file.write(sdata + "f,")
        endstr = "\n" + "}" + ";" + "\n" + "\n"
        file.write(endstr)

    def save_fcparam(self, file, splitname, model_names):
        weightname = splitname + ".weight"
        biasname = splitname + ".bias"
        weight = self.net.state_dict()[weightname]

        w_name = splitname.upper().replace(".", "_") + "_WEIGHT"  # FC_WEIGHT
        w_begin = "const float " + w_name + "[] = " + "\n" + "{" + "\n"
        file.write(w_begin)

        weidata = weight.data.cpu().numpy()
        ochannel, ichannel = weidata.shape
        kg4 = " "
        dimstr = "//    " + str(ochannel) + ", " + str(ichannel) + ", " + "\n"
        file.write(dimstr)
        #####save weight
        for l in range(ochannel):
            ffdata = weidata[l, :]
            for i in range(ichannel):
                fdata = ffdata[i]
                file.write(kg4)
                if fdata >= 0:
                    sdata = ('%.6f' % fdata)
                    file.write("+" + sdata + "f,")
                if fdata < 0:
                    sdata = ('%.6f' % fdata)
                    file.write(sdata + "f,")
            file.write("\n")
        endstr = "}" + ";" + "\n" + "\n"
        file.write(endstr)

        if biasname in model_names:
            b_name = splitname.upper().replace(".", "_") + "_BIAS"  # FC_BIAS
            b_begin = "const float " + b_name + "[] = " + "\n" + "{" + "\n"
            file.write(b_begin)
            bias = self.net.state_dict()[biasname]
            biasdata = bias.data.cpu().numpy()
            dimstr1 = "//    " + str(ochannel) + "," + "\n"
            file.write(dimstr1)
            #####save bias
            for v in range(int(ochannel)):
                bdata = biasdata[v]
                file.write(kg4)
                if bdata >= 0:
                    sdata = ('%.6f' % bdata)
                    file.write("+" + sdata + "f,")
                if bdata < 0:
                    sdata = ('%.6f' % bdata)
                    file.write(sdata + "f,")
            file.write("\n")
            endstr = "}" + ";" + "\n" + "\n"
            file.write(endstr)

    def forward(self, cfg_path, src_path):
        param_cfg = open(cfg_path, 'w+')
        param_src = open(src_path, 'w+')
        head_name = "#include " + "\"" + src_path + "\"" + "\n" + "\n"
        param_cfg.write(head_name)

        # 保存模型所有层的名称
        model_names = []
        for name in self.net.state_dict():
            name = name.strip()
            model_names.append(name)

        for name, parameters in self.net.named_parameters():
            name = name.strip()
            if name.endswith("weight"):
                pre = name.split(".weight")[0]
            if name.endswith("bias"):
                pre = name.split(".bias")[0]
            param_dim = parameters.ndim

            self.save_param_name_pytorch(param_cfg, name, param_dim, model_names)
            if param_dim == 2 and name.endswith("weight"):  # fc weights
                self.save_fcparam(param_src, pre, model_names)
            if param_dim == 4 and name.endswith("weight"):  # conv weights
                self.save_convparam_(param_src, pre, model_names)
            if param_dim == 1 and name.endswith("weight"):  # bn weights
                weightname = pre + ".weight"
                biasname = pre + ".bias"
                meaname = pre + ".running_mean"
                varname = pre + ".running_var"
                param_weight = self.net.state_dict()[weightname]
                param_bias = self.net.state_dict()[biasname]
                param_mean = self.net.state_dict()[meaname]
                param_var = self.net.state_dict()[varname]
                self.save_bnparam(param_src, pre, param_weight, param_bias, param_mean, param_var)
        param_cfg.close()
        param_src.close()

def save_feature_channel(txtpath, feaMap, batch, channel, height, width):
    file = open(txtpath, 'w+')
    if batch > 1 or batch < 1 or channel < 1:
        print("feature map more than 1 batch will not save")
    if batch ==1:#4维
        feaMap = feaMap.squeeze(0)
        feadata = feaMap.data.cpu().numpy()
        if height > 0 and width > 0:
            for i in range(channel):
                file.write("channel --> " + str(i) + "\n")
                for j in range(height):
                    for k in range(width):
                        fdata = feadata[i, j, k]
                        if fdata >= 0:
                            sdata = ('%.6f' % fdata)
                            file.write("+" + sdata + ",")
                        if fdata < 0:
                            sdata = ('%.6f' % fdata)
                            file.write(sdata + ",")
                    file.write("\n")
                file.write("\n")
        if height < 1 and width < 1:#2维
            for i in range(channel):
                file.write("channel --> " + str(i) + "\n")
                fdata = feadata[i]
                if fdata >= 0:
                    sdata = ('%.6f' % fdata)
                    file.write("+" + sdata + ",")
                if fdata < 0:
                    sdata = ('%.6f' % fdata)
                    file.write(sdata + ",")
                file.write("\n")
        if height > 0 and width < 1:#3维
            for i in range(channel):
                file.write("channel --> " + str(i) + "\n")
                for j in range(height):
                    fdata = feadata[i, j]
                    if fdata >= 0:
                        sdata = ('%.6f' % fdata)
                        file.write("+" + sdata + ",")
                    if fdata < 0:
                        sdata = ('%.6f' % fdata)
                        file.write(sdata + ",")
                file.write("\n")
    file.close()