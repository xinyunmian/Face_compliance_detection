import torch
import os
import torch.utils.data as data
from load_data import Shadow_DataLoader, Shadow_collate
from slim_net import NeckShadowNet
from myconfig import config
from loss import NeckShadowLoss
#cuda
torch.cuda.set_device(0)

def adjust_learning_rate(epoch, optimizer):
    lr = config.lr
    if epoch > 650:
        lr = lr / 1000000
    elif epoch > 600:
        lr = lr / 100000
    elif epoch > 550:
        lr = lr / 10000
    elif epoch > 350:
        lr = lr / 1000
    elif epoch > 200:
        lr = lr / 100
    elif epoch > 80:
        lr = lr / 10
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def train():
    # 读取训练测试数据
    data_train = Shadow_DataLoader(config.train_data, config.train_txt)
    train_loader = data.DataLoader(data_train, batch_size=config.batch_size, shuffle=True, num_workers=0, collate_fn=Shadow_collate)
    train_len = len(data_train)
    max_batch = train_len // config.batch_size

    # 定义网络结构
    SkinNet = NeckShadowNet()
    # pretrained_path = config.resetNet  # 需要修改
    # pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    # SkinNet.load_state_dict(pretrained_dict)
    SkinNet.cuda()

    criterion = NeckShadowLoss()
    optimizer = torch.optim.SGD(SkinNet.parameters(), lr=config.lr, weight_decay=config.weight_decay, momentum=0.9)
    SkinNet.train()
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
            out = SkinNet(images)
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
            torch.save(SkinNet.state_dict(), config.model_save + "/" + "NeckShadow_{}.pth".format(epoch))

        # 打印度量
        print("Epoch {}, TrainLoss: {}".format(epoch, loss.item()))
    torch.save(SkinNet.state_dict(), config.model_save + "/" + "NeckShadow_final.pth")

if __name__ == "__main__":
    train()

















