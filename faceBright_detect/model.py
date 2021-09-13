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

class faceBrightHead(nn.Module):
    def __init__(self, inchannel=128, outchannel=100):
        super(faceBrightHead, self).__init__()
        self.inc = inchannel
        self.outc = outchannel
        self.conv1x1 = nn.Conv2d(self.inc, self.outc, 1, 1, 0, bias=False)
    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.size(0), -1)

class FaceBright(nn.Module):
    def __init__(self, n_score=100):
        super(FaceBright, self).__init__()
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

        self.conv_dark = conv_dw(128, 128, 1)
        self.darkhead = faceBrightHead(128, n_score // 4)

        self.conv_bright = conv_dw(128, 128, 1)
        self.brighthead = faceBrightHead(128, n_score // 4)

        self.conv_yy = conv_dw(128, 128, 1)
        self.yyhead = faceBrightHead(128, n_score // 4)

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
        x11 = x

        x_dark = self.conv_dark(x11)
        x_dark = self.darkhead(x_dark)
        x_dark = F.softmax(x_dark, dim=1)

        x_bright = self.conv_bright(x11)
        x_bright = self.brighthead(x_bright)
        x_bright = F.softmax(x_bright, dim=1)

        x_yy = self.conv_yy(x11)
        x_yy = self.yyhead(x_yy)
        x_yy = F.softmax(x_yy, dim=1)

        return x_dark, x_bright, x_yy


class FaceBrightNet(nn.Module):
    def __init__(self, n_class=4):
        super(FaceBrightNet, self).__init__()
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

        self.conv_dark = conv_dw(128, 128, 1)
        self.fc_dark = nn.Linear(in_features=2 * 2 * 128, out_features=n_class)

        self.conv_bright = conv_dw(128, 128, 1)
        self.fc_bright = nn.Linear(in_features=2 * 2 * 128, out_features=n_class)

        self.conv_yy = conv_dw(128, 128, 1)
        self.fc_yy = nn.Linear(in_features=2 * 2 * 128, out_features=n_class)

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
        x11 = x

        x_dark = self.conv_dark(x11)
        x_dark = x_dark.view(x_dark.size(0), -1)
        x_dark = self.fc_dark(x_dark)

        x_bright = self.conv_bright(x11)
        x_bright = x_bright.view(x_bright.size(0), -1)
        x_bright = self.fc_bright(x_bright)

        x_yy = self.conv_yy(x11)
        x_yy = x_yy.view(x_yy.size(0), -1)
        x_yy = self.fc_yy(x_yy)

        return x_dark, x_bright, x_yy

if __name__ == "__main__":
    import time
    from load_data import pytorch_to_dpcoreParams
    net = FaceBrightNet(n_class=4)

    _weight = "weights/better_210.pth"  # 需要修改
    _dict = torch.load(_weight, map_location=lambda storage, loc: storage)
    net.load_state_dict(_dict)
    net.eval()
    saveparams = pytorch_to_dpcoreParams(net)
    saveparams.forward("faceLight_param_cfg.h", "faceLight_param_src.h")

    # torch.save(net.state_dict(), 'facebright.pth')
    x = torch.randn(1, 3, 128, 128)
    load_t0 = time.time()
    y = net(x)
    load_t1 = time.time()
    forward_time = load_t1 - load_t0
    print("前向传播时间:{:.4f}秒".format(forward_time))
    print(y.size())














