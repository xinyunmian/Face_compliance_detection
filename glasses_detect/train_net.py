import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from load_data import Face_classification
from mask_net import eyeGlassNet
from myconfig import config
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
    elif epoch > 420:
        lr = lr / 10000
    elif epoch > 380:
        lr = lr / 1000
    elif epoch > 180:
        lr = lr / 100
    elif epoch > 80:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def test(net, test_data):
    net.eval()
    test_acc = 0.0
    total = 0.0
    for i, (images, labels) in enumerate(test_data):
        images = images.cuda()
        labels = labels.cuda()
        # Predict classes using images from the test set
        outputs = net(images)
        _, prediction = torch.max(outputs.data, 1)
        total += labels.size(0)
        test_acc += (labels == prediction).sum().item()
        # test_acc += torch.sum(prediction == labels.data)
    # Compute the average acc and loss over all 10000 test images
    test_acc = test_acc / total
    return test_acc

def train():
    # 读取训练测试数据
    data_train = Face_classification(config.train_data, config.train_txt)
    data_test = Face_classification(config.val_data, config.val_txt)
    train_loader = data.DataLoader(data_train, batch_size=config.batch_size, shuffle=True, num_workers=0)
    test_loader = data.DataLoader(data_test, batch_size=config.batch_size, shuffle=True, num_workers=0)
    train_len = len(data_train)
    max_batch = train_len // config.batch_size

    # 定义网络结构
    myNet = eyeGlassNet(n_class=config.num_classes)
    if config.pretrain:
        d_dict = torch.load(config.pre_weight, map_location=lambda storage, loc: storage)
        myNet.load_state_dict(d_dict)
        print("load weight done !!!")
    myNet.cuda()
    myNet.train()

    # 梯度优化以及loss
    optimizer = optim.SGD(myNet.parameters(), lr=config.lr, weight_decay=config.weight_decay, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss(reduction='sum')

    best_acc = 0.0
    for epoch in range(config.start_epoch, config.epochs):
        print("Epoch {}".format(epoch))
        print('I am training, please wait...')
        batch = 0
        train_acc = 0.0
        train_loss = 0.0
        lr_cur = 0.01
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda()
            labels = labels.cuda()
            # 清除所有累积梯度
            optimizer.zero_grad()
            # 用来自测试集的图像预测类
            outputs = myNet(images)
            # 根据实际标签和预测值计算损失
            loss = loss_fn(outputs + 1e-8, labels)
            batch_loss = loss.cpu().item()
            train_loss += batch_loss
            _, prediction = torch.max(outputs.data, 1)
            batch_true_num = (labels == prediction).sum().item()
            batch_acc = batch_true_num / images.size(0)
            train_acc += batch_true_num
            # 传播损失
            loss.backward()
            # 根据计算的梯度调整参数
            optimizer.step()
            batch += 1

            for params in optimizer.param_groups:
                lr_cur = params['lr']
            if batch % 30 == 0:
                print("Epoch: %d/%d || batch:%d/%d batch_loss: %.3f || batch_acc: %.2f || LR: %.6f"
                      % (epoch, config.epochs, batch, max_batch, batch_loss, batch_acc, lr_cur))
        # 调用学习率调整函数
        adjust_learning_rate(epoch, optimizer)
        # 计算模型在50000张训练图像上的准确率和损失值
        train_acc = train_acc / train_len
        train_loss = train_loss / train_len
        # 用测试集评估
        test_acc = test(myNet, test_loader)
        # 若测试准确率高于当前最高准确率，则保存模型
        if test_acc > best_acc:
            torch.save(myNet.state_dict(), config.model_save + "/" + "glasses_better{}.pth".format(epoch))
            best_acc = test_acc

        if (epoch % 2 == 0 and epoch > 0):
            torch.save(myNet.state_dict(), config.model_save + "/" + "glasses_{}.pth".format(epoch))

        # 打印度量
        print("Epoch{}, Train Accuracy:{} , Test Accuracy:{}".format(epoch, train_acc, test_acc))

    torch.save(myNet.state_dict(), config.model_save + "/" + "glasses_final.pth")
# train()
if __name__ == '__main__':
    train()