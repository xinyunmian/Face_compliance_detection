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

class BeautyDetectNet(nn.Module):
    def __init__(self, channel_list6=[16, 24, 32, 48, 64, 96]):
        super(BeautyDetectNet, self).__init__()
        self.conv1 = conv_dw(3, channel_list6[0], 2)
        self.conv2 = conv_dw(channel_list6[0], channel_list6[1], 2)
        self.conv3 = conv_dw(channel_list6[1], channel_list6[1], 1)
        self.conv4 = conv_dw(channel_list6[1], channel_list6[2], 2)
        self.conv5 = conv_dw(channel_list6[2], channel_list6[2], 1)
        self.conv6 = conv_dw(channel_list6[2], channel_list6[3], 2)
        self.conv7 = conv_dw(channel_list6[3], channel_list6[3], 1)
        self.conv8 = conv_dw(channel_list6[3], channel_list6[4], 2)
        self.conv9 = conv_dw(channel_list6[4], channel_list6[4])
        self.conv10 = conv_dw(channel_list6[4], channel_list6[5], 2)
        self.conv11 = conv_dw(channel_list6[5], channel_list6[5], 1)
        self.fc = nn.Linear(in_features=2 * 2 * 96, out_features=1)

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
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x


if __name__ == "__main__":
    import time
    from load_data import pytorch_to_dpcoreParams
    net = BeautyDetectNet(n_class=2)
    # torch.save(net.state_dict(), 'net.pth')
    x = torch.randn(1, 3, 128, 128)
    load_t0 = time.time()
    y = net(x)
    load_t1 = time.time()
    forward_time = load_t1 - load_t0
    print("前向传播时间:{:.4f}秒".format(forward_time))
    print(y.size())














