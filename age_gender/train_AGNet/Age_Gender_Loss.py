import torch
import torch.nn as nn
import torch.nn.functional as F

class Age_GenderLoss(nn.Module):
    def __init__(self):
        super(Age_GenderLoss, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(2), requires_grad=True).cuda()
        self.l1_loss = nn.SmoothL1Loss()
        self.entropy_loss = nn.CrossEntropyLoss()

    def forward(self, age_pre, gender_pre, age_label, gender_label):
        age_label = age_label.unsqueeze(1).float()
        loss_age = self.l1_loss(age_pre, age_label)
        loss_gender = self.entropy_loss(gender_pre, gender_label)
        var = torch.exp(-self.log_vars)
        loss_sum = (loss_age * var[0] + self.log_vars[0]) + (loss_gender * var[1] + self.log_vars[1])

        return loss_age, loss_gender, loss_sum


class AGDLDL_Loss(nn.Module):
    def __init__(self):
        super(AGDLDL_Loss, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(3), requires_grad=True).cuda()
        self.kl_loss = nn.KLDivLoss(reduction='none')
        self.l1_loss = nn.SmoothL1Loss()
        self.entropy_loss = nn.CrossEntropyLoss()
        self.rank = torch.Tensor([i for i in range(116)]).cuda()

    def forward(self, age_pre, gender_pre, age_label, normage_label, gender_label):
        preages = torch.sum(age_pre * self.rank, dim=1).unsqueeze(1)
        age_label = age_label.unsqueeze(1).float()
        loss_age = self.l1_loss(preages, age_label)

        log_age = torch.log(age_pre)
        loss_logage = self.kl_loss(log_age, normage_label)
        loss_normage = loss_logage.sum()/loss_logage.shape[0]

        loss_gender = self.entropy_loss(gender_pre, gender_label)
        var = torch.exp(-self.log_vars)
        loss_sum = (loss_age * var[0] + self.log_vars[0]) + (loss_normage * var[1] + self.log_vars[1]) + (loss_gender * var[2] + self.log_vars[2])

        return loss_age, loss_normage, loss_gender, loss_sum