import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class mobilev3_Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(mobilev3_Block, self).__init__()
        self.stride = stride
        self.se = semodule


        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        convt = self.conv1(x)
        bnt = self.bn1(convt)
        out = self.nolinear1(bnt)
        # save_feature_channel("fm_txt/py/backbone0_ep_conv.txt", convt, 1, 16, 64, 64)
        # save_feature_channel("fm_txt/py/backbone0_ep_bn.txt", bnt, 1, 16, 64, 64)
        # save_feature_channel("fm_txt/py/backbone0_ep_act.txt", out, 1, 16, 64, 64)
        # out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class mobilev3_AgeGenderNet(nn.Module):
    def __init__(self):
        super(mobilev3_AgeGenderNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.backbone = nn.Sequential(
            mobilev3_Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
            mobilev3_Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
            mobilev3_Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
            mobilev3_Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
            mobilev3_Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            mobilev3_Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            mobilev3_Block(5, 40, 120, 48, hswish(), SeModule(48), 2),
            mobilev3_Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
            mobilev3_Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
            mobilev3_Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
            mobilev3_Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
            mobilev3_Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        )

        self.fc_age = nn.Linear(in_features=2 * 2 * 96, out_features=1)
        self.age_norm = nn.Sigmoid()

        self.fc_gender = nn.Linear(in_features=2 * 2 * 96, out_features = 2)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out_conv1 = self.hs1(self.bn1(self.conv1(x)))
        x1 = self.backbone(out_conv1)
        x1 = x1.view(x1.size(0), -1)

        out_gender = self.fc_gender(x1)

        out_age = self.fc_age(x1)
        out_age = self.age_norm(out_age)

        return out_age, out_gender

class mobilev3_AGDLDL(nn.Module):
    def __init__(self):
        super(mobilev3_AGDLDL, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.backbone = nn.Sequential(
            mobilev3_Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
            mobilev3_Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
            mobilev3_Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
            mobilev3_Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
            mobilev3_Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            mobilev3_Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            mobilev3_Block(5, 40, 120, 48, hswish(), SeModule(48), 2),
            mobilev3_Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
            mobilev3_Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
            mobilev3_Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
            mobilev3_Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
            mobilev3_Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        )

        self.fc_age = nn.Linear(in_features=2 * 2 * 96, out_features=116)
        self.age_norm = nn.Sigmoid()

        self.fc_gender = nn.Linear(in_features=2 * 2 * 96, out_features = 2)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out_conv1 = self.hs1(self.bn1(self.conv1(x)))
        x1 = self.backbone(out_conv1)
        x1 = x1.view(x1.size(0), -1)

        out_gender = self.fc_gender(x1)

        out_age = self.fc_age(x1)
        out_age = self.age_norm(out_age)
        out_age = F.normalize(out_age, p=1, dim=1)

        return out_age, out_gender

class mobilev3_AGDLDL_new(nn.Module):
    def __init__(self):
        super(mobilev3_AGDLDL_new, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        # self.backbone0 = mobilev3_Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2)
        # self.backbone1 = mobilev3_Block(3, 16, 32, 24, nn.ReLU(inplace=True), None, 2)
        # self.backbone2 = mobilev3_Block(3, 24, 32, 24, nn.ReLU(inplace=True), None, 1)
        # self.backbone3 = mobilev3_Block(5, 24, 64, 40, hswish(), SeModule(40), 2)
        # self.backbone4 = mobilev3_Block(5, 40, 64, 40, hswish(), SeModule(40), 1)
        # self.backbone5 = mobilev3_Block(5, 40, 96, 40, hswish(), SeModule(40), 1)
        # self.backbone6 = mobilev3_Block(5, 40, 96, 48, hswish(), SeModule(48), 2)
        # self.backbone7 = mobilev3_Block(5, 48, 128, 48, hswish(), SeModule(48), 1)
        # self.backbone8 = mobilev3_Block(5, 48, 128, 48, hswish(), SeModule(48), 1)
        # self.backbone9 = mobilev3_Block(5, 48, 256, 96, hswish(), SeModule(96), 2)
        # self.backbone10 = mobilev3_Block(5, 96, 256, 96, hswish(), SeModule(96), 1)
        # self.backbone11 = mobilev3_Block(5, 96, 512, 96, hswish(), SeModule(96), 1)

        self.backbone = nn.Sequential(
            mobilev3_Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
            mobilev3_Block(3, 16, 32, 24, nn.ReLU(inplace=True), None, 2),
            mobilev3_Block(3, 24, 32, 24, nn.ReLU(inplace=True), None, 1),
            mobilev3_Block(5, 24, 64, 40, hswish(), SeModule(40), 2),
            mobilev3_Block(5, 40, 64, 40, hswish(), SeModule(40), 1),
            mobilev3_Block(5, 40, 96, 40, hswish(), SeModule(40), 1),
            mobilev3_Block(5, 40, 96, 48, hswish(), SeModule(48), 2),
            mobilev3_Block(5, 48, 128, 48, hswish(), SeModule(48), 1),
            mobilev3_Block(5, 48, 128, 48, hswish(), SeModule(48), 1),
            mobilev3_Block(5, 48, 256, 96, hswish(), SeModule(96), 2),
            mobilev3_Block(5, 96, 256, 96, hswish(), SeModule(96), 1),
            mobilev3_Block(5, 96, 512, 96, hswish(), SeModule(96), 1),
        )
        self.conv2 = nn.Conv2d(96, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.hs2 = hswish()

        self.fc_age = nn.Linear(in_features=2 * 2 * 256, out_features=116)
        self.age_norm = nn.Sigmoid()

        self.fc_gender = nn.Linear(in_features=2 * 2 * 256, out_features = 2)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out_conv1 = self.hs1(self.bn1(self.conv1(x)))
        x1 = self.backbone(out_conv1)
        x1 = self.hs2(self.bn2(self.conv2(x1)))
        x1 = x1.view(x1.size(0), -1)

        out_gender = self.fc_gender(x1)

        out_age = self.fc_age(x1)
        out_age = self.age_norm(out_age)
        out_age = F.normalize(out_age, p=1, dim=1)

        # conv1 = self.conv1(x)
        # bn1 = self.bn1(conv1)
        # hsw1 = self.hs1(bn1)
        #
        # backbone0 = self.backbone[0](hsw1)
        # backbone1 = self.backbone[1](backbone0)
        # backbone2 = self.backbone[2](backbone1)
        # backbone3 = self.backbone[3](backbone2)
        # backbone4 = self.backbone[4](backbone3)
        # backbone5 = self.backbone[5](backbone4)
        # backbone6 = self.backbone[6](backbone5)
        # backbone7 = self.backbone[7](backbone6)
        # backbone8 = self.backbone[8](backbone7)
        # backbone9 = self.backbone[9](backbone8)
        # backbone10 = self.backbone[10](backbone9)
        # backbone11 = self.backbone[11](backbone10)
        #
        # conv2 = self.conv2(backbone11)
        # bn2 = self.bn2(conv2)
        # hsw2 = self.hs2(bn2)
        #
        # viewout = hsw2.view(hsw2.size(0), -1)
        #
        # fc_age = self.fc_age(viewout)
        # age_sig = self.age_norm(fc_age)
        # out_age = F.normalize(age_sig, p=1, dim=1)
        #
        # out_gender = self.fc_gender(viewout)

        # save_feature_channel("fm_txt/py/conv1.txt", conv1, 1, 16, 64, 64)
        # save_feature_channel("fm_txt/py/bn1.txt", bn1, 1, 16, 64, 64)
        # save_feature_channel("fm_txt/py/hsw1.txt", hsw1, 1, 16, 64, 64)
        #
        # save_feature_channel("fm_txt/py/backbone0.txt", backbone0, 1, 16, 32, 32)
        # save_feature_channel("fm_txt/py/backbone1.txt", backbone1, 1, 24, 16, 16)
        # save_feature_channel("fm_txt/py/backbone2.txt", backbone2, 1, 24, 16, 16)
        # save_feature_channel("fm_txt/py/backbone3.txt", backbone3, 1, 40, 8, 8)
        # save_feature_channel("fm_txt/py/backbone4.txt", backbone4, 1, 40, 8, 8)
        # save_feature_channel("fm_txt/py/backbone5.txt", backbone5, 1, 40, 8, 8)
        # save_feature_channel("fm_txt/py/backbone6.txt", backbone6, 1, 48, 4, 4)
        # save_feature_channel("fm_txt/py/backbone7.txt", backbone7, 1, 48, 4, 4)
        # save_feature_channel("fm_txt/py/backbone8.txt", backbone8, 1, 48, 4, 4)
        # save_feature_channel("fm_txt/py/backbone9.txt", backbone9, 1, 96, 2, 2)
        # save_feature_channel("fm_txt/py/backbone10.txt", backbone10, 1, 96, 2, 2)
        # save_feature_channel("fm_txt/py/backbone11.txt", backbone11, 1, 96, 2, 2)
        #
        # save_feature_channel("fm_txt/py/conv2.txt", conv2, 1, 256, 2, 2)
        # save_feature_channel("fm_txt/py/bn2.txt", bn2, 1, 256, 2, 2)
        # save_feature_channel("fm_txt/py/hsw2.txt", hsw2, 1, 256, 2, 2)
        #
        # #age
        # save_feature_channel("fm_txt/py/fc_age.txt", fc_age, 1, 116, 0, 0)
        # save_feature_channel("fm_txt/py/age_sig.txt", age_sig, 1, 116, 0, 0)
        #
        # #gender
        # save_feature_channel("fm_txt/py/fc_gender.txt", out_gender, 1, 2, 0, 0)

        return out_age, out_gender

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


if __name__ == "__main__":
    net = mobilev3_AGDLDL_new()
    net.eval()
    net.load_state_dict(torch.load("weights/AgeGender_0713.pth", map_location=lambda storage, loc: storage), strict=True)
    # torch.save(net.state_dict(), 'mobilev3_AgeGenderNet.pth')
    x = torch.rand(1, 3, 128, 128)
    ageout, genderout = net(x)
    print(ageout.data)
    print(ageout.size())