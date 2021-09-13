import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from load_data import LoadDataBeauty, Beauty_collate
from slim_net import BeautyDetectNet
from myconfig import config
from loss import BeautyLoss
import math
from collections import OrderedDict

#cuda
torch.cuda.set_device(0)

#学习率变化
def adjust_learning_rate(epoch, optimizer):
    lr = config.lr

    if epoch > 650:
        lr = lr / 1000000
    elif epoch > 500:
        lr = lr / 100000
    elif epoch > 180:
        lr = lr / 10000
    elif epoch > 100:
        lr = lr / 1000
    elif epoch > 50:
        lr = lr / 100
    elif epoch > 10:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def train():
    # 读取训练测试数据
    data_train = LoadDataBeauty(config.train_data, config.train_txt)
    train_loader = data.DataLoader(data_train, batch_size=config.batch_size, shuffle=True, num_workers=0, collate_fn=Beauty_collate)
    train_len = len(data_train)
    max_batch = train_len // config.batch_size

    # 定义网络结构
    beautyNet = BeautyDetectNet()
    # pretrained_path = config.resetNet  # 需要修改
    # pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    # beautyNet.load_state_dict(pretrained_dict)
    beautyNet.cuda()

    criterion = BeautyLoss()
    optimizer = torch.optim.SGD(beautyNet.parameters(), lr=config.lr, weight_decay=config.weight_decay, momentum=0.9)
    beautyNet.train()
    small_loss = 100000
    for epoch in range(config.epochs):
        print("Epoch {}".format(epoch))
        print('I am training, please wait...')
        batch = 0
        lr_cur = 0.01
        for i, (images, label) in enumerate(train_loader):
            batch += 1
            # 若GPU可用，将图像和标签移往GPU
            images = images.cuda()
            label = label.unsqueeze(1).cuda()
            # 预测
            out = beautyNet(images)
            # 清除所有累积梯度
            optimizer.zero_grad()

            loss = criterion(out, label)

            loss.backward()
            optimizer.step()

            for params in optimizer.param_groups:
                lr_cur = params['lr']

            if batch % 5 == 0:
                print("Epoch:{}/{} || Epochiter: {}/{} || loss: {:.4f} || LR: {:.8f}".format(epoch, config.epochs, max_batch, batch, loss.item(), lr_cur))

        # 调用学习率调整函数
        adjust_learning_rate(epoch, optimizer)

        if (epoch % 2 == 0 and epoch > 0):
            torch.save(beautyNet.state_dict(), config.model_save + "/" + "Beauty_{}.pth".format(epoch), _use_new_zipfile_serialization=False)

        # 打印度量
        print("Epoch {}, TrainLoss: {}".format(epoch, loss.item()))

        if (loss.item() < small_loss):
            small_loss = loss.item()
            torch.save(beautyNet.state_dict(), config.model_save + "/" + "Beauty_better.pth", _use_new_zipfile_serialization=False)

if __name__ == "__main__":
    train()