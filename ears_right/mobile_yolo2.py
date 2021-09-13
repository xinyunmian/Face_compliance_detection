import torch.nn as nn
import torch.nn.functional as F
import torch

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

class YoloHead(nn.Module):
    def __init__(self, inchannel=128, outchannel=35):
        super(YoloHead, self).__init__()
        self.inc = inchannel
        self.outc = outchannel
        self.conv1x1 = nn.Conv2d(self.inc, self.outc, 1, 1, 0, bias=False)
    def forward(self, x):
        out = self.conv1x1(x)
        return out

class yolo_mobile(nn.Module):
    def __init__(self, nclass=2, nanchors=5):
        super(yolo_mobile, self).__init__()
        self.outc = (5 + nclass) * nanchors
        self.conv1 = conv_bn(3, 16, 2)
        self.conv2 = conv_dw(16, 16, 1)
        self.conv3 = conv_dw(16, 32, 2)
        self.conv4 = conv_dw(32, 32, 1)
        self.conv5 = conv_dw(32, 48, 2)
        self.conv6 = conv_dw(48, 48, 1)
        self.conv7 = conv_dw(48, 64, 2)
        self.conv8 = conv_dw(64, 64, 1)
        self.conv9 = conv_dw(64, 64, 1)
        self.conv10 = conv_dw(64, 128, 2)
        self.conv11 = conv_dw(128, 128, 1)
        self.conv12 = conv_dw(128, 128, 1)

        self.Yhead = YoloHead(inchannel=128, outchannel=self.outc)

        self.lconv = conv_dw(128, 64, 2)
        self.l_type = nn.Linear(in_features=5 * 5 * 64, out_features=1)
        self.l_norm = nn.Sigmoid()

        self.rconv = conv_dw(128, 64, 2)
        self.r_type = nn.Linear(in_features=5 * 5 * 64, out_features=1)
        self.r_norm = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # b, c, h, w = x.shape
        # save_feature_channel("txt/p/conv2.txt", x, b, c, h, w)
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

        out_yolo = self.Yhead(x)

        xl = self.lconv(x)
        xl = xl.view(xl.size(0), -1)
        l_type = self.l_type(xl)
        l_type = self.l_norm(l_type)

        xr = self.rconv(x)
        xr = xr.view(xr.size(0), -1)
        r_type = self.r_type(xr)
        r_type = self.r_norm(r_type)

        return out_yolo, l_type, r_type

class yolo_type(nn.Module):
    def __init__(self, nclass=2, nregress=1, nanchors=5):
        super(yolo_type, self).__init__()
        self.outc = (5 + nclass + nregress) * nanchors
        # self.yoloc = 5 + nclass
        # self.allc = 5 + nclass + nregress
        # self.nanchors = nanchors
        self.conv1 = conv_bn(3, 16, 2)
        self.conv2 = conv_dw(16, 16, 1)
        self.conv3 = conv_dw(16, 32, 2)
        self.conv4 = conv_dw(32, 32, 1)
        self.conv5 = conv_dw(32, 48, 2)
        self.conv6 = conv_dw(48, 48, 1)
        self.conv7 = conv_dw(48, 64, 2)
        self.conv8 = conv_dw(64, 64, 1)
        self.conv9 = conv_dw(64, 64, 1)
        self.conv10 = conv_dw(64, 128, 2)
        self.conv11 = conv_dw(128, 128, 1)
        self.conv12 = conv_dw(128, 128, 1)

        self.Yhead = YoloHead(inchannel=128, outchannel=self.outc)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # b, c, h, w = x.shape
        # save_feature_channel("txt/p/conv2.txt", x, b, c, h, w)
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

        out = self.Yhead(x)
        # out = out.view(out.size(0), self.allc, self.nanchors, 10, 10)
        # yolo_out = out[:, 0:self.yoloc, :, :, :].view(out.size(0), self.yoloc * self.nanchors, 10, 10)
        # ltype = out[:, self.yoloc, :, :, :]
        # rtype = out[:, self.yoloc + 1, :, :, :]

        return out   #yolo_out, ltype, rtype

if __name__ == "__main__":
    import time
    net = yolo_type(nclass=2, nregress=1, nanchors=5)

    # glass_weight = "weights/eye_glasses_300.pth"  # 需要修改
    # glass_dict = torch.load(glass_weight, map_location=lambda storage, loc: storage)
    # net.load_state_dict(glass_dict)
    # net.eval()
    # saveparams = pytorch_to_dpcoreParams(net)
    # saveparams.forward("glass_param_cfg.h", "glass_param_src.h")

    torch.save(net.state_dict(), 'yolo.pth')
    x = torch.randn(1, 3, 320, 320)
    yolo, lt, rt = net(x)
    print(yolo.size())













