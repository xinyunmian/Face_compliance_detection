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

class mobilev3_AgeGenderNet(nn.Module):
    def __init__(self):
        super(mobilev3_AgeGenderNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        # self.backbone = nn.Sequential(
        #     mobilev3_Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
        #     mobilev3_Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
        #     mobilev3_Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
        #     mobilev3_Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
        #     mobilev3_Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
        #     mobilev3_Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
        #     mobilev3_Block(5, 40, 120, 48, hswish(), SeModule(48), 2),
        #     mobilev3_Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
        #     mobilev3_Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
        #     mobilev3_Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
        #     mobilev3_Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        #     mobilev3_Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        # )

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

        self.fc_age = nn.Linear(in_features=2 * 2 * 256, out_features=1)
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

        return out_age, out_gender

class mobilev3_AGDLDL(nn.Module):
    def __init__(self):
        super(mobilev3_AGDLDL, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        # self.backbone = nn.Sequential(
        #     mobilev3_Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
        #     mobilev3_Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
        #     mobilev3_Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
        #     mobilev3_Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
        #     mobilev3_Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
        #     mobilev3_Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
        #     mobilev3_Block(5, 40, 120, 48, hswish(), SeModule(48), 2),
        #     mobilev3_Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
        #     mobilev3_Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
        #     mobilev3_Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
        #     mobilev3_Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        #     mobilev3_Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        # )

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

        return out_age, out_gender

if __name__ == "__main__":
    net = mobilev3_AgeGenderNet()
    net.eval()
    torch.save(net.state_dict(), 'mobilev3_AgeGenderNet.pth')
    x = torch.randn(1, 3, 128, 128)
    ageout, genderout = net(x)
    print(ageout.size())