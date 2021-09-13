import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from load_data import pytorch_to_dpcoreParams, save_feature_channel

class conv_bn(nn.Module):
    def __init__(self, inp, oup, stride = 1):
        super(conv_bn, self).__init__()
        self.conv = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class conv_dw(nn.Module):
    def __init__(self, inp, oup, stride = 1):
        super(conv_dw, self).__init__()
        self.conv1 = nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)
        self.bn1 = nn.BatchNorm2d(inp)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.conv2 = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(oup)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class FaceQualitySlim(nn.Module):
    def __init__(self, channels=[16, 32, 48, 64, 128]):
        super(FaceQualitySlim, self).__init__()
        self.conv1 = conv_dw(3, channels[0], 2)
        self.conv2 = conv_dw(channels[0], channels[0], 1)
        self.conv3 = conv_dw(channels[0], channels[0], 1)
        self.conv4 = conv_dw(channels[0], channels[1], 2)
        self.conv5 = conv_dw(channels[1], channels[1], 1)
        self.conv6 = conv_dw(channels[1], channels[1], 1)
        self.conv7 = conv_dw(channels[1], channels[2], 2)
        self.conv8 = conv_dw(channels[2], channels[2], 1)
        self.conv9 = conv_dw(channels[2], channels[2], 1)
        self.conv10 = conv_dw(channels[2], channels[3], 2)
        self.conv11 = conv_dw(channels[3], channels[3], 1)
        self.conv12 = conv_dw(channels[3], channels[3], 1)
        self.conv13 = conv_dw(channels[3], channels[4], 2)
        self.conv14 = conv_dw(channels[4], channels[4], 1)
        self.conv15 = conv_dw(channels[4], channels[4], 1)

        self.fc_out = nn.Linear(in_features=channels[4] * 3 * 3, out_features=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)

        x = x.view(x.size(0), -1)
        out = self.fc_out(x)
        return out

class LDA_Module(nn.Module):
    def __init__(self, inc=32, ouc=16):
        super(LDA_Module, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(inc, inc, 1, 1, 0, bias=False)
        self.fc = nn.Linear(in_features=inc * 1 * 1, out_features=ouc)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class TargetModule(nn.Module):
    def __init__(self, inc=128, ldac=64, ouc=1):
        super(TargetModule, self).__init__()
        self.fc1 = nn.Linear(in_features=inc, out_features=3 * ldac)
        self.fc2 = nn.Linear(in_features=3 * ldac, out_features=2 * ldac)
        self.fc3 = nn.Linear(in_features=2 * ldac, out_features=ldac)
        self.fc_out = nn.Linear(in_features=ldac, out_features=ouc)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc_out(x)
        return x

class FaceQualityNet(nn.Module):
    def __init__(self, channels=[16, 32, 48, 64, 128], lda_outc=32):
        super(FaceQualityNet, self).__init__()
        self.conv1 = conv_dw(3, channels[0], 2)
        self.conv2 = conv_dw(channels[0], channels[0], 1)
        self.conv3 = conv_dw(channels[0], channels[0], 1)
        self.lda1 = LDA_Module(inc=channels[0], ouc=lda_outc)

        self.conv4 = conv_dw(channels[0], channels[1], 2)
        self.conv5 = conv_dw(channels[1], channels[1], 1)
        self.conv6 = conv_dw(channels[1], channels[1], 1)
        self.lda2 = LDA_Module(inc=channels[1], ouc=lda_outc)

        self.conv7 = conv_dw(channels[1], channels[2], 2)
        self.conv8 = conv_dw(channels[2], channels[2], 1)
        self.conv9 = conv_dw(channels[2], channels[2], 1)
        self.lda3 = LDA_Module(inc=channels[2], ouc=lda_outc)

        self.conv10 = conv_dw(channels[2], channels[3], 2)
        self.conv11 = conv_dw(channels[3], channels[3], 1)
        self.conv12 = conv_dw(channels[3], channels[3], 1)
        self.lda4 = LDA_Module(inc=channels[3], ouc=lda_outc)

        self.conv13 = conv_dw(channels[3], channels[4], 2)
        self.conv14 = conv_dw(channels[4], channels[4], 1)
        self.conv15 = conv_dw(channels[4], channels[4], 1)
        self.lda5 = LDA_Module(inc=channels[4], ouc=lda_outc)

        self.target = TargetModule(inc=5 * lda_outc, ldac=lda_outc, ouc=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        lda1_out = self.lda1(x)
        # b, c, h, w = x.shape
        # save_feature_channel("txt/conv3p.txt", x, b, c, h, w)

        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        lda2_out = self.lda2(x)

        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        lda3_out = self.lda3(x)

        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        lda4_out = self.lda4(x)

        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        lda5_out = self.lda5(x)

        x = torch.cat((lda1_out, lda2_out, lda3_out, lda4_out, lda5_out), 1)
        out = self.target(x)
        return out

if __name__ == "__main__":
    net = FaceQualityNet()

    # _weight = "weights/FaceSkin_Mobile_300.pth"  # 需要修改
    # _dict = torch.load(_weight, map_location=lambda storage, loc: storage)
    # net.load_state_dict(_dict)
    # saveparams = pytorch_to_dpcoreParams(net)
    # saveparams.forward("faceSkin_param_cfg.h", "faceSkin_param_src.h")

    torch.save(net.state_dict(), 'face_quality.pth')
    net.eval()
    x = torch.randn(4, 3, 757, 891)
    y = net(x)
    print(y.size())






















