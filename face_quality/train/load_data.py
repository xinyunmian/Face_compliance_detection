import numpy as np
import torch
import torch.utils.data as data
import cv2
from myconfig import config
import random
import math

def get_patches(img, patch_size=32, patch_num=16):
    patches_img = []
    imgh, imgw, imgc = img.shape
    if(imgw <=patch_size or imgh <= patch_size):
        return patches_img
    for i in range(patch_num):
        x = np.random.randint(0, imgw - patch_size)
        y = np.random.randint(0, imgh - patch_size)
        patch_img = img[y:y+patch_size, x:x+patch_size, :]
        patches_img.append(patch_img)
    return patches_img

def get_patches_augment(img, patch_size=96, timenum=15):
    patches_img = []
    imgh, imgw, imgc = img.shape
    if(imgw <=patch_size or imgh <= patch_size):
        patch_num = 0
        return patches_img, patch_num

    maxwh = max(imgh, imgw)
    patch_num = int((maxwh / patch_size) * timenum)
    for i in range(patch_num):
        x = np.random.randint(0, imgw - patch_size)
        y = np.random.randint(0, imgh - patch_size)
        patch_img = img[y:y+patch_size, x:x+patch_size, :]
        patches_img.append(patch_img)
    return patches_img, patch_num

def mirror_face(img):
    mir_img = img[:, ::-1]
    return mir_img

class Data_augment(object):
    def __init__(self, aug, mir):
        self.aug = aug
        self.mir = mir
    def __call__(self, image):
        if self.aug > 0:
            if self.mir > 0:
                image = mirror_face(image)
        return image

class FaceQuality_DataLoader(data.Dataset):
    def __init__(self, dir_path, txt_path, patch_size=96, patch_num=8):
        with open(txt_path, 'r') as f:
            self.imgs = list(map(lambda line: line.strip().split(' '), f))
        self.dir_path = dir_path
        self.patchSize = patch_size
        self.patchNum = patch_num

    def __getitem__(self, index):
        path, quality_score = self.imgs[index]
        img_path = self.dir_path + "/" + path
        img = cv2.imread(img_path)
        # image augment
        if img is None:
            print(self.imgs[index])

        imgpatches = get_patches(img, patch_size=self.patchSize, patch_num=self.patchNum)
        augment_patches = torch.FloatTensor(self.patchNum, 3, self.patchSize, self.patchSize)
        score = np.zeros(self.patchNum, dtype=float)

        for i in range(self.patchNum):
            onepatch = imgpatches[i]
            augment_ornot = random.choice([0, 1])
            mirror_ornot = random.choice([0, 1, 2])
            process = Data_augment(augment_ornot, mirror_ornot)
            onepatch = process(onepatch)
            onepatch = onepatch.astype(np.float32)
            # img = (img - config.bgr_mean) / config.bgr_std
            onepatch = onepatch / 255.0
            onepatch = onepatch.transpose(2, 0, 1)
            onepatch = torch.from_numpy(onepatch)
            augment_patches[i, :, :, :] = onepatch
            quality_score = int(quality_score)
            score[i] = self.label_change(quality_score)

        return augment_patches, score

    def label_change(self, lab):
        change_lab = 0.0
        if lab == 0:
            change_lab = 0.0
        if lab == 1:
            change_lab = 0.25
        if lab == 2:
            change_lab = 0.5
        if lab == 3:
            change_lab = 0.75
        if lab == 4:
            change_lab = 1.0
        return change_lab

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)

def quality_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for batchid, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            else:
                annos = torch.tensor(tup).float()
                targets.append(annos)
    return (torch.cat(imgs, 0), torch.cat(targets, 0))

class FaceQuality_augment(data.Dataset):
    def __init__(self, dir_path, txt_path, patch_size=96, scale=15):
        with open(txt_path, 'r') as f:
            self.imgs = list(map(lambda line: line.strip().split(' '), f))
        self.dir_path = dir_path
        self.patchSize = patch_size
        self.scale = scale

    def __getitem__(self, index):
        path, quality_score = self.imgs[index]
        img_path = self.dir_path + "/" + path
        img = cv2.imread(img_path)
        # image augment
        if img is None:
            print(self.imgs[index])

        # imgpatches = get_patches(img, patch_size=self.patchSize, patch_num=self.patchNum)
        imgpatches, num_patch = get_patches_augment(img, patch_size=self.patchSize, timenum=self.scale)
        augment_patches = torch.FloatTensor(num_patch, 3, self.patchSize, self.patchSize)
        score = np.zeros(num_patch, dtype=float)

        for i in range(num_patch):
            onepatch = imgpatches[i]
            augment_ornot = random.choice([0, 1])
            mirror_ornot = random.choice([0, 1, 2])
            process = Data_augment(augment_ornot, mirror_ornot)
            onepatch = process(onepatch)
            onepatch = onepatch.astype(np.float32)
            # img = (img - config.bgr_mean) / config.bgr_std
            onepatch = onepatch / 255.0
            onepatch = onepatch.transpose(2, 0, 1)
            onepatch = torch.from_numpy(onepatch)
            augment_patches[i, :, :, :] = onepatch
            quality_score = int(quality_score)
            score[i] = self.label_change(quality_score)

        return augment_patches, score

    def label_change(self, lab):
        change_lab = 0.0
        if lab == 0:
            change_lab = 0.0
        if lab == 1:
            change_lab = 0.25
        if lab == 2:
            change_lab = 0.5
        if lab == 3:
            change_lab = 0.75
        if lab == 4:
            change_lab = 1.0
        return change_lab

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)

def quality_augment_collate(batch):
    targets = []
    imgs = []
    num = 0
    for batchid, sample in enumerate(batch):
        num += 1
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            else:
                lenpatch = len(tup)
                lab_score = torch.zeros(lenpatch, 2).float()
                lab_score[:, 0] = batchid
                annos = torch.tensor(tup).float()
                lab_score[:, 1] = annos
                targets.append(lab_score)
    return (torch.cat(imgs, 0), torch.cat(targets, 0), num)

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