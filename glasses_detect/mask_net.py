import torch.nn as nn
import torch.nn.functional as F
import torch
import math

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

class eyeGlassNet(nn.Module):
    def __init__(self, n_class=4):
        super(eyeGlassNet, self).__init__()
        self.conv1 = conv_bn(3, 16, 2)
        self.conv2 = conv_dw(16, 16, 1)
        self.conv3 = conv_dw(16, 32, 2)
        self.conv4 = conv_dw(32, 32, 1)
        self.conv5 = conv_dw(32, 48, 2)
        self.conv6 = conv_dw(48, 48, 1)
        self.conv7 = conv_dw(48, 64, 2)
        self.conv8 = conv_dw(64, 64, 1)
        self.conv9 = conv_dw(64, 96, 2)
        self.conv10 = conv_dw(96, 96, 1)
        self.conv11 = conv_dw(96, 128, 2)

        self.fc = nn.Linear(in_features=2 * 1 * 128, out_features=n_class)

    def forward(self, x):
        x = self.conv1(x)
        # b, c, h, w = x.shape
        # save_feature_channel("txt/p/conv2.txt", x, b, c, h, w)
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
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    import time
    from load_data import pytorch_to_dpcoreParams
    net = eyeGlassNet(n_class=4)
    glass_weight = "weights/glasses_300_0827.pth"  # 需要修改
    glass_dict = torch.load(glass_weight, map_location=lambda storage, loc: storage)
    net.load_state_dict(glass_dict)
    net.eval()
    # torch.save(net.state_dict(), "weights/pretrain.pth", _use_new_zipfile_serialization=False)

    # for name in net.state_dict():
    #     print(name)
    #     name = name.strip()
    #     parameter = net.state_dict()[name]

    saveparams = pytorch_to_dpcoreParams(net)
    saveparams.forward("glass_param_cfg.h", "glass_param_src.h")
    # torch.save(net.state_dict(), 'mobilev3_small.pth')
    x = torch.randn(1, 3, 64, 128)
    load_t0 = time.time()
    y = net(x)
    load_t1 = time.time()
    forward_time = load_t1 - load_t0
    print("前向传播时间:{:.4f}秒".format(forward_time))
    print(y.size())














