import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from load_data import Face_classification
from slim_net import Slim
from myconfig import config
import math
from collections import OrderedDict
#cuda
cuda_avail = torch.cuda.is_available()
#定义网络结构
myNet = Slim(nclasses = config.num_classes)
print('reset net weights...')
myNet.load_state_dict(torch.load(config.resetNet))

# print('reset net weights...')
# state_dict = torch.load(config.resetNet)
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     head = k[:7]
#     if head == 'module.':
#         name = k[7:] # remove `module.`
#     else:
#         name = k
#     new_state_dict[name] = v
# myNet.load_state_dict(new_state_dict)

#读取训练测试数据
data_train = Face_classification(config.train_data, config.train_txt)
data_test = Face_classification(config.val_data, config.val_txt)
train_loader = data.DataLoader(data_train,batch_size=config.batch_size,shuffle=True,num_workers=0)
test_loader = data.DataLoader(data_test,batch_size=config.batch_size,shuffle=False,num_workers=0)
train_len = len(data_train)
test_len = len(data_test)

if cuda_avail:
    myNet.cuda()

#梯度优化以及loss
optimizer = optim.Adam(myNet.parameters(), lr=config.lr, weight_decay=config.weight_decay)
loss_fn = nn.CrossEntropyLoss()

#学习率变化
def adjust_learning_rate(epoch):
    lr = 0.001

    if epoch > 650:
        lr = lr / 1000000
    elif epoch > 500:
        lr = lr / 100000
    elif epoch > 320:
        lr = lr / 10000
    elif epoch > 250:
        lr = lr / 1000
    elif epoch > 150:
        lr = lr / 100
    elif epoch > 80:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def test():
    myNet.eval()
    test_acc = 0.0
    total = 0.0
    for i, (images, labels) in enumerate(test_loader):
        if cuda_avail:
            images = images.cuda()
            labels = labels.cuda()
        # Predict classes using images from the test set
        outputs = myNet(images)
        _, prediction = torch.max(outputs.data, 1)
        total += labels.size(0)
        test_acc += (labels == prediction).sum().item()
        # test_acc += torch.sum(prediction == labels.data)
    # Compute the average acc and loss over all 10000 test images
    test_acc = test_acc / total
    return test_acc

# def train():
#     best_acc = 0.0
#     for epoch in range(config.epochs):
#         print("Epoch {}".format(epoch))
#         print('I am training, please wait...')
#         myNet.train()
#         train_acc = 0.0
#         train_loss = 0.0
#         for i, (images, labels) in enumerate(train_loader):
#             # 若GPU可用，将图像和标签移往GPU
#             if cuda_avail:
#                 images = images.cuda()
#                 labels = labels.cuda()
#             # 清除所有累积梯度
#             optimizer.zero_grad()
#             # 用来自测试集的图像预测类
#             outputs = myNet(images)
#             # 根据实际标签和预测值计算损失
#             loss = loss_fn(outputs, labels)
#             # 传播损失
#             loss.backward()
#             # 根据计算的梯度调整参数
#             optimizer.step()
#             train_loss += loss.cpu().item() * images.size(0)
#             _, prediction = torch.max(outputs.data, 1)
#             # train_acc += torch.sum(prediction == labels.data)
#             train_acc += (labels == prediction).sum().item()
#         # 调用学习率调整函数
#         adjust_learning_rate(epoch)
#         # 计算模型在50000张训练图像上的准确率和损失值
#         train_acc = train_acc / train_len
#         train_loss = train_loss / train_len
#         # 用测试集评估
#         test_acc = test()
#         # torch.save(myNet.state_dict(), config.model_save + "/" + "model_{}.pth".format(epoch))
#         # 若测试准确率高于当前最高准确率，则保存模型
#         if test_acc > best_acc:
#             torch.save(myNet.state_dict(), config.model_save + "/" + "htmodel_{}.pth".format(epoch))
#             best_acc = test_acc
#
#         if (epoch % 5 == 0 and epoch > 0):
#             torch.save(myNet.state_dict(), config.model_save + "/" + "huoti_{}.pth".format(epoch))
#
#         # 打印度量
#         print("Epoch {}, Train Accuracy: {} , TrainLoss: {} , Test Accuracy: {}".format(epoch, train_acc, train_loss, test_acc))
#
#     torch.save(myNet.state_dict(), config.model_save + "/" + "ht_final_{}.pth".format(epoch))

max_batch = train_len // config.batch_size
def train_batch():
    best_acc = 0.0
    for epoch in range(config.epochs):
        print("Epoch {}".format(epoch))
        print('I am training, please wait...')
        batch = 0
        myNet.train()
        sumacc = 0.0
        sumloss = 0.0
        train_acc = 0.0
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # 若GPU可用，将图像和标签移往GPU
            if cuda_avail:
                images = images.cuda()
                labels = labels.cuda()
            # 清除所有累积梯度
            optimizer.zero_grad()
            # 用来自测试集的图像预测类
            outputs = myNet(images)
            # 根据实际标签和预测值计算损失
            loss = loss_fn(outputs, labels)
            train_loss = loss.cpu().item() * images.size(0)
            sumloss += train_loss
            _, prediction = torch.max(outputs.data, 1)
            # train_acc += torch.sum(prediction == labels.data)
            train_correct = (labels == prediction).sum().item()
            train_acc = train_correct / images.size(0)
            sumacc += train_correct
            # 传播损失
            loss.backward()
            # 根据计算的梯度调整参数
            optimizer.step()
            batch += 1
            print("Epoch: %d/%d || batch:%d/%d loss: %.3f || train_acc: %.2f"
                  % (epoch, config.epochs, batch, max_batch, train_loss, train_acc))

        # 调用学习率调整函数
        adjust_learning_rate(epoch)
        sumacc = sumacc / train_len
        sumloss = sumloss / train_len
        # 若测试准确率高于当前最高准确率，则保存模型
        if sumacc > best_acc:
            torch.save(myNet.state_dict(), config.model_save + "/" + "htmodel_{}.pth".format(epoch))
            best_acc = sumacc

        if (epoch % 5 == 0 and epoch > 0):
            torch.save(myNet.state_dict(), config.model_save + "/" + "huoti_{}.pth".format(epoch))

        # 打印度量
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {}".format(epoch, sumacc, sumloss))

    torch.save(myNet.state_dict(), config.model_save + "/" + "ht_final_{}.pth".format(epoch))

# train()
if __name__ == '__main__':
    train_batch()