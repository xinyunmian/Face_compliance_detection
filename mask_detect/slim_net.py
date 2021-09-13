import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class conv_bn(nn.Module):
    def __init__(self, inp, oup, stride = 1):
        super(conv_bn, self).__init__()
        self.conv = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class conv2d_depth(nn.Module):
    def __init__(self, inp, oup, kernel=1, stride=1, pad=0):
        super(conv2d_depth, self).__init__()
        self.conv1 = nn.Conv2d(inp, inp, kernel_size = kernel, stride = stride, padding=pad, groups=inp)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inp, oup, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

class conv_dw(nn.Module):
    def __init__(self, inp, oup, stride = 1):
        super(conv_dw, self).__init__()
        self.conv1 = nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)
        self.bn1 = nn.BatchNorm2d(inp)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(oup)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class maskNet(nn.Module):
    def __init__(self, n_class=2):
        super(maskNet, self).__init__()
        self.conv1 = conv_bn(3, 16, 2)
        self.conv2 = conv_dw(16, 32, 1)
        self.conv3 = conv_dw(32, 32, 2)
        self.conv4 = conv_dw(32, 32, 1)
        self.conv5 = conv_dw(32, 64, 2)
        self.conv6 = conv_dw(64, 64, 1)
        self.conv7 = conv_dw(64, 128, 2)
        self.conv8 = conv_dw(128, 128, 2)

        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),
            nn.ReLU(inplace=True),
            conv2d_depth(64, 128, kernel=3, stride=2, pad=1),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(in_features=2 * 2 * 128, out_features=n_class)

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

        # x1 = x[:, :, 0:2, 0:2].contiguous()
        # x2 = x[:, :, 1:3, 0:2].contiguous()
        # x3 = x[:, :, 0:2, 1:3].contiguous()
        # x4 = x[:, :, 1:3, 1:3].contiguous()
        # x1 = x1.view(x1.size(0), -1)
        # x2 = x2.view(x2.size(0), -1)
        # x3 = x3.view(x3.size(0), -1)
        # x4 = x4.view(x4.size(0), -1)
        # x1 = self.fc(x1)
        # x2 = self.fc(x2)
        # x3 = self.fc(x3)
        # x4 = self.fc(x4)
        #
        # prediction11 = torch.softmax(x1.data, dim=1)
        # prediction22 = torch.softmax(x2.data, dim=1)
        # prediction33 = torch.softmax(x3.data, dim=1)
        # prediction44 = torch.softmax(x4.data, dim=1)
        # _, prediction1 = torch.max(x1.data, 1)
        # _, prediction2 = torch.max(x2.data, 1)
        # _, prediction3 = torch.max(x3.data, 1)
        # _, prediction4 = torch.max(x4.data, 1)
        #
        # prediction = prediction1 + prediction2 + prediction3 + prediction4
        # passornot = 0
        # if prediction > 2:
        #     passornot = 1
        #
        # return passornot


        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    from save_params import pytorch_to_dpcoreParams
    import time
    net = maskNet(n_class=2)
    slim_weight = torch.load("weights/mask_450.pth")
    net.load_state_dict(slim_weight)
    net.eval()

    saveparams = pytorch_to_dpcoreParams(net)
    saveparams.forward("maskNet_param_cfg.h", "maskNet_param_src.h")
    print("done")











