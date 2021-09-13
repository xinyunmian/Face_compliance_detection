import torch
import torch.nn as nn
import torch.nn.functional as F

class FaceQualityLoss(nn.Module):
    def __init__(self):
        super(FaceQualityLoss, self).__init__()
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

class FaceQuality_PatchLoss(nn.Module):
    def __init__(self, train_batch=32):
        super(FaceQuality_PatchLoss, self).__init__()
        self.train_batch = train_batch
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, output, target):
        output = torch.sigmoid(output)
        sum_loss = 0
        for i in range(self.train_batch):
            batchid = float(i)
            maskbatch = (target[:, 0] == batchid)
            out = output[maskbatch]
            mean_out = torch.mean(out)
            lab = target[maskbatch]
            lab = lab[:, 1]
            mean_lab = torch.mean(lab)
            loss = self.loss(mean_out, mean_lab)
            sum_loss += loss
        return sum_loss


