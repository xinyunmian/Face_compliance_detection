import torch.nn as nn
import torch.nn.functional as F
import torch
import math

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

class NeckShadowNet(nn.Module):
    def __init__(self):
        super(NeckShadowNet, self).__init__()
        self.conv1 = conv_dw(3, 16, 2)
        self.conv2 = conv_dw(16, 16, 1)
        self.conv3 = conv_dw(16, 16, 1)
        self.conv4 = conv_dw(16, 24, 2)
        self.conv5 = conv_dw(24, 24, 1)
        self.conv6 = conv_dw(24, 24, 1)
        self.conv7 = conv_dw(24, 32, 2)
        self.conv8 = conv_dw(32, 32, 1)
        self.conv9 = conv_dw(32, 32, 1)
        self.conv10 = conv_dw(32, 48, 2)
        self.conv11 = conv_dw(48, 48, 1)
        self.conv12 = conv_dw(48, 48, 1)
        self.conv13 = conv_dw(48, 64, 2)
        self.conv14 = conv_dw(64, 64, 1)
        self.conv15 = conv_dw(64, 64, 1)

        self.fc_out = nn.Linear(in_features=4 * 3 * 64, out_features=1)

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

if __name__ == "__main__":
    from load_data import pytorch_to_dpcoreParams
    net = NeckShadowNet()

    # _weight = "weights/FaceSkin_Mobile_300.pth"  # 需要修改
    # _dict = torch.load(_weight, map_location=lambda storage, loc: storage)
    # net.load_state_dict(_dict)
    # saveparams = pytorch_to_dpcoreParams(net)
    # saveparams.forward("faceSkin_param_cfg.h", "faceSkin_param_src.h")

    # torch.save(net.state_dict(), 'facebright.pth')
    net.eval()
    x = torch.randn(16, 3, 96, 96)
    y = net(x)
    print(y.size())






















