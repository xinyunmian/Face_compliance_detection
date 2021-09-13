import torch
import torch.nn as nn
import torch.nn.functional as F

class PoseLoss(nn.Module):
    def __init__(self):
        super(PoseLoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, output, target):
        """
        :param output: batch * 1
        :param target: batch * 1
        :return: loss
        """
        output = torch.sigmoid(output)
        loss = self.loss(output, target)
        return loss