import torch
import os
import torch.utils.data as data
from load_data import faceBright_DataLoader
from model import FaceBrightNet
from myconfig import config
from loss import facebright_loss
#cuda
torch.cuda.set_device(0)

#读取训练测试数据
data_train = faceBright_DataLoader(config.train_data, config.train_txt)
train_loader = data.DataLoader(data_train,batch_size=config.batch_size,shuffle=True,num_workers=0)
train_len = len(data_train)

#定义网络结构
FBNet = FaceBrightNet()
FBNet.cuda()

criterion = facebright_loss()
optimizer = torch.optim.SGD(FBNet.parameters(), lr=config.lr, weight_decay=config.weight_decay, momentum=0.9)

def adjust_learning_rate(epoch):
    lr = config.lr
    if epoch > 650:
        lr = lr / 1000000
    elif epoch > 500:
        lr = lr / 100000
    elif epoch > 300:
        lr = lr / 10000
    elif epoch > 200:
        lr = lr / 1000
    elif epoch > 100:
        lr = lr / 100
    elif epoch > 30:
        lr = lr / 10
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

max_batch = train_len // config.batch_size
def trainDLDL():
    for epoch in range(config.epochs):
        print("Epoch {}".format(epoch))
        print('I am training, please wait...')
        FBNet.train()
        batch = 0
        lr_cur = 0.01
        for i, (images, darklab, darknorm, brightlab, brightnorm, yylab, yynorm) in enumerate(train_loader):
            batch += 1
            # 若GPU可用，将图像和标签移往GPU
            images = images.cuda()
            darklab = darklab.cuda()
            darknorm = darknorm.cuda()
            brightlab = brightlab.cuda()
            brightnorm = brightnorm.cuda()
            yylab = yylab.cuda()
            yynorm = yynorm.cuda()
            # 预测
            pre_dark, pre_bright, pre_yy = FBNet(images)
            # 清除所有累积梯度
            optimizer.zero_grad()

            loss_d, loss_b, loss_y, loss = criterion(pre_dark, pre_bright, pre_yy, darklab,
                                                      brightlab, yylab, darknorm, brightnorm, yynorm)
            loss.backward()
            optimizer.step()

            for params in optimizer.param_groups:
                lr_cur = params['lr']
            print("Epoch:{}/{} || Epochiter: {}/{} || Dark: {:.4f} Bright: {:.4f} YinYang: {:.4f} Sum: {:.4f} || LR: {:.8f}"
                  .format(epoch,config.epochs,max_batch,batch,loss_d.item(),loss_b.item(),loss_y.item(),loss.item(),lr_cur))

        # 调用学习率调整函数
        adjust_learning_rate(epoch)
        torch.save(FBNet.state_dict(), config.model_save + "/" + "faceBright_final.pth")

        if (epoch % 2 == 0 and epoch > 0):
            torch.save(FBNet.state_dict(), config.model_save + "/" + "faceBright_{}.pth".format(epoch))

def train():
    best_acc = 0.0
    for epoch in range(config.epochs):
        print("Epoch {}".format(epoch))
        print('I am training, please wait...')
        FBNet.train()
        batch = 0
        lr_cur = 0.01
        train_acc = 0.0
        true_sumnum = 0
        for i, (images, darklab, brightlab, yylab) in enumerate(train_loader):
            batch += 1
            # 若GPU可用，将图像和标签移往GPU
            images = images.cuda()
            darklab = darklab.cuda()
            brightlab = brightlab.cuda()
            yylab = yylab.cuda()
            # 预测
            pre_dark, pre_bright, pre_yy = FBNet(images)
            # 清除所有累积梯度
            optimizer.zero_grad()

            loss_d, loss_b, loss_y, loss = criterion(pre_dark, pre_bright, pre_yy, darklab,
                                                      brightlab, yylab)

            _, max_dark = torch.max(pre_dark.data, 1)
            _, max_bright = torch.max(pre_bright.data, 1)
            _, max_yy = torch.max(pre_yy.data, 1)
            batch_size = images.size(0)

            right_num = 0
            for k in range(batch_size):
                if (max_dark[k]==darklab[k] and max_bright[k]==brightlab[k] and max_yy[k]==yylab[k]):
                    right_num += 1
            batch_acc = right_num / batch_size
            true_sumnum += right_num


            loss.backward()
            optimizer.step()

            for params in optimizer.param_groups:
                lr_cur = params['lr']

            if batch % 20 == 0:
                print("Epoch:{}/{} || Epochiter: {}/{} || Dark: {:.4f} Bright: {:.4f} YinYang: {:.4f} Sum: {:.4f} Acc: {:.4f} || LR: {:.8f}"
                  .format(epoch,config.epochs,max_batch,batch,loss_d.item(),loss_b.item(),loss_y.item(),loss.item(),batch_acc,lr_cur))

        # 调用学习率调整函数
        adjust_learning_rate(epoch)

        train_acc = true_sumnum / train_len
        if train_acc > best_acc:
            torch.save(FBNet.state_dict(), config.model_save + "/" + "better_{}.pth".format(epoch))
            best_acc = train_acc

        if (epoch % 2 == 0 and epoch > 0):
            torch.save(FBNet.state_dict(), config.model_save + "/" + "faceBright_{}.pth".format(epoch))

        # 打印度量
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {}".format(epoch, train_acc, loss.item()))
    torch.save(FBNet.state_dict(), config.model_save + "/" + "faceBright_final.pth")

if __name__ == "__main__":
    # trainDLDL()
    train()

















