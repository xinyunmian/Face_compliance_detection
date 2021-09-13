import torch
import torch.nn as nn
import torch.nn.functional as F

class facebright_loss(nn.Module):
    def __init__(self):
        super(facebright_loss, self).__init__()
        self.l_dark = nn.CrossEntropyLoss(reduction='sum')
        self.l_bright = nn.CrossEntropyLoss(reduction='sum')
        self.l_yy = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, dark_pre, bright_pre, yy_pre, dark_label, bright_label, yy_label):
        # 过暗
        loss_dark = self.l_dark(dark_pre, dark_label)
        # 过亮
        loss_bright = self.l_bright(bright_pre, bright_label)
        # 阴阳脸
        loss_yy = self.l_yy(yy_pre, yy_label)

        loss_sum = loss_dark + loss_bright + loss_yy

        return loss_dark, loss_bright, loss_yy, loss_sum


class fb_loss(nn.Module):
    def __init__(self):
        super(fb_loss, self).__init__()
        self.kl_dark = nn.KLDivLoss(reduction='mean')
        self.l1_dark = nn.SmoothL1Loss(reduction='mean')
        self.kl_bright = nn.KLDivLoss(reduction='mean')
        self.l1_bright = nn.SmoothL1Loss(reduction='mean')
        self.kl_yy = nn.KLDivLoss(reduction='mean')
        self.l1_yy = nn.SmoothL1Loss(reduction='mean')
        self.rank = torch.Tensor([i for i in range(100)]).cuda()

    def forward(self, dark_pre, bright_pre, yy_pre, dark, bright, yy, dark_label, bright_label, yy_label):
        # 过暗
        predark = torch.sum(dark_pre * self.rank, dim=1).unsqueeze(1)
        dark = dark.unsqueeze(1).float()
        loss_dark1 = self.l1_dark(predark, dark)
        log_dark = torch.log(dark_pre)
        loss_dark2 = self.kl_dark(log_dark, dark_label)
        loss_dark = loss_dark1 + loss_dark2

        # 过亮
        prebright = torch.sum(bright_pre * self.rank, dim=1).unsqueeze(1)
        bright = bright.unsqueeze(1).float()
        loss_bright1 = self.l1_bright(prebright, bright)
        log_bright = torch.log(bright_pre)
        loss_bright2 = self.kl_bright(log_bright, bright_label)
        loss_bright = loss_bright1 + loss_bright2

        # 阴阳脸
        preyy = torch.sum(yy_pre * self.rank, dim=1).unsqueeze(1)
        yy = yy.unsqueeze(1).float()
        loss_yy1 = self.l1_yy(preyy, yy)
        log_yy = torch.log(yy_pre)
        loss_yy2 = self.kl_yy(log_yy, yy_label)
        loss_yy = loss_yy1 + loss_yy2

        loss_sum = loss_dark + loss_bright + loss_yy

        return loss_dark, loss_bright, loss_yy, loss_sum


class FaceSkinLoss(nn.Module):
    def __init__(self):
        super(FaceSkinLoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def get_target_mask(self, target):
        nB = target.size(0)
        nC = target.size(1)
        mask = torch.zeros(nB, nC)
        for b in range(nB):
            bright = target[b][0]
            dark = target[b][1]
            yinyang = target[b][2]
            skin = target[b][3]
            ratio = target[b][4]
            if bright >= 0:
                mask[b][0] = 1
            if dark >= 0:
                mask[b][1] = 1
            if yinyang >= 0:
                mask[b][2] = 1
            if skin >= 0:
                mask[b][3] = 1
            if ratio >= 0:
                mask[b][4] = 1
        return mask

    def forward(self, output, target):
        """
        :param output: batch * 5
        :param target: batch * 5
        :return: loss
        """
        output = torch.sigmoid(output)
        cls_mask = self.get_target_mask(target)
        cls_mask = cls_mask.cuda()
        loss = self.loss(output * cls_mask, target * cls_mask)
        return loss