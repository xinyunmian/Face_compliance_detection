import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

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
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class ClassHead(nn.Module):
    def __init__(self, inchannels=256, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)
    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self, inchannels=256, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)
    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self, inchannels=256, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0)
    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 10)

class FPN(nn.Module):
    def __init__(self,in_channels,out_channel):
        super(FPN,self).__init__()
        self.out1 = nn.Sequential(
            nn.Conv2d(in_channels[0], out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        self.out2 = nn.Sequential(
            nn.Conv2d(in_channels[1], out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        self.out3 = nn.Sequential(
            nn.Conv2d(in_channels[2], out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        self.out4 = nn.Sequential(
            nn.Conv2d(in_channels[3], out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        self.merge1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        self.merge2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        self.merge3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
    def forward(self, inputs):
        output1 = self.out1(inputs[0])
        output2 = self.out2(inputs[1])
        output3 = self.out3(inputs[2])
        output4 = self.out4(inputs[3])

        up4 = F.interpolate(output4, size=[output3.size(2), output3.size(3)], mode="nearest")
        output33 = output3 + up4
        output33 = self.merge3(output33)

        up3 = F.interpolate(output33, size=[output2.size(2), output2.size(3)], mode="nearest")
        output22 = output2 + up3
        output22 = self.merge2(output22)

        up2 = F.interpolate(output22, size=[output1.size(2), output1.size(3)], mode="nearest")
        output11 = output1 + up2
        output11 = self.merge1(output11)

        out = [output11, output22, output33, output4]
        return out

class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        self.conv1_norelu = nn.Sequential(
            nn.Conv2d(in_channel, out_channel // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel // 2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel // 4),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        self.conv2_norelu = nn.Sequential(
            nn.Conv2d(out_channel // 4, out_channel // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel // 4),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channel // 4, out_channel // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel // 4),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        self.conv3_norelu = nn.Sequential(
            nn.Conv2d(out_channel // 4, out_channel // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel // 4),
        )

    def forward(self, input):
        conv1_no = self.conv1_norelu(input)

        conv2_out = self.conv2(input)
        conv2_no = self.conv2_norelu(conv2_out)

        conv3_out = self.conv3(conv2_out)
        conv3_no = self.conv3_norelu(conv3_out)

        concat_out = torch.cat([conv1_no, conv2_no, conv3_no], dim=1)
        out = F.relu(concat_out)
        return out

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

class mobilev3Fpn_small(nn.Module):
    def __init__(self, cfg = None):
        super(mobilev3Fpn_small, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        #__init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride)
        self.stage1 = nn.Sequential(
            mobilev3_Block(3, 16, 32, 16, nn.ReLU(inplace=True), SeModule(16), 2),
            mobilev3_Block(3, 16, 64, 32, nn.ReLU(inplace=True), None, 2),
            mobilev3_Block(3, 32, 64, 32, nn.ReLU(inplace=True), None, 1),
        )

        self.stage2 = nn.Sequential(
            mobilev3_Block(5, 32, 64, 48, hswish(), SeModule(48), 2),
            mobilev3_Block(5, 48, 128, 48, hswish(), SeModule(48), 1),
            mobilev3_Block(5, 48, 128, 48, hswish(), SeModule(48), 1),
        )

        self.stage3 = nn.Sequential(
            mobilev3_Block(5, 48, 128, 64, hswish(), SeModule(64), 2),
            mobilev3_Block(5, 64, 256, 64, hswish(), SeModule(64), 1),
            mobilev3_Block(5, 64, 256, 64, hswish(), SeModule(64), 1),
        )

        self.stage4 = nn.Sequential(
            mobilev3_Block(5, 64, 256, 128, hswish(), SeModule(128), 2),
            mobilev3_Block(5, 128, 512, 128, hswish(), SeModule(128), 1),
            mobilev3_Block(5, 128, 512, 128, hswish(), SeModule(128), 1),
        )

        self.anchor = cfg['min_sizes']
        in_channel_list = [32, 48, 64, 128]
        out_channel = 64
        self.fpn = FPN(in_channel_list, out_channel)
        self.ssh1 = SSH(out_channel, out_channel)
        self.ssh2 = SSH(out_channel, out_channel)
        self.ssh3 = SSH(out_channel, out_channel)
        self.ssh4 = SSH(out_channel, out_channel)
        self.ClassHead = self._make_class_head(out_channel)
        self.BboxHead = self._make_bbox_head(out_channel)
        self.LandmarkHead = self._make_landmark_head(out_channel)

    def _make_class_head(self, input=64):
        classhead = nn.ModuleList()
        classhead.append(ClassHead(input, len(self.anchor[0])))
        classhead.append(ClassHead(input, len(self.anchor[1])))
        classhead.append(ClassHead(input, len(self.anchor[2])))
        classhead.append(ClassHead(input, len(self.anchor[3])))
        return classhead

    def _make_bbox_head(self, input=64):
        bboxhead = nn.ModuleList()
        bboxhead.append(BboxHead(input, len(self.anchor[0])))
        bboxhead.append(BboxHead(input, len(self.anchor[1])))
        bboxhead.append(BboxHead(input, len(self.anchor[2])))
        bboxhead.append(BboxHead(input, len(self.anchor[3])))
        return bboxhead

    def _make_landmark_head(self, input=64):
        landmarkhead = nn.ModuleList()
        landmarkhead.append(LandmarkHead(input, len(self.anchor[0])))
        landmarkhead.append(LandmarkHead(input, len(self.anchor[1])))
        landmarkhead.append(LandmarkHead(input, len(self.anchor[2])))
        landmarkhead.append(LandmarkHead(input, len(self.anchor[3])))
        return landmarkhead

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
        x1 = self.stage1(out_conv1)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        xes = [x1, x2, x3, x4]
        # FPN
        fpn = self.fpn(xes)

        # SSH
        ssh1 = self.ssh1(fpn[0])
        ssh2 = self.ssh2(fpn[1])
        ssh3 = self.ssh3(fpn[2])
        ssh4 = self.ssh4(fpn[3])

        features = [ssh1, ssh2, ssh3, ssh4]
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        # output = (bbox_regressions, classifications, ldm_regressions)#train
        output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)#test

        return output

if __name__ == "__main__":
    from retinaface.config import cfg_mnet
    import time
    net = mobilev3Fpn_small(cfg=cfg_mnet)
    net.eval()
    # torch.save(net.state_dict(), 'mobilev3_small.pth')
    x = torch.randn(1, 3, 640, 640)
    load_t0 = time.time()
    y = net(x)
    load_t1 = time.time()
    forward_time = load_t1 - load_t0
    print("前向传播时间:{:.4f}秒".format(forward_time))
    print(y[0].size())















































