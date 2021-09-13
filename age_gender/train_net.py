import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.utils.data as data
from load_data import Face_classification
from slim_net import maskNet
from myconfig import config
from collections import OrderedDict
#cuda
torch.cuda.set_device(1)
#定义网络结构
myNet = maskNet(n_class = config.num_classes)

print('reset net weights...')
state_dict = torch.load(config.resetNet)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    head = k[:7]
    if head == 'module.':
        name = k[7:] # remove `module.`
    else:
        name = k
    new_state_dict[name] = v
myNet.load_state_dict(new_state_dict)

#读取训练测试数据
data_train = Face_classification(config.train_data, config.train_txt)
train_loader = data.DataLoader(data_train,batch_size=config.batch_size,shuffle=True,num_workers=0)
train_len = len(data_train)
#test_loader = data.DataLoader(data_test,batch_size=config.batch_size,shuffle=False,num_workers=0)
#data_test = Face_classification(config.val_data, config.val_txt)
#test_len = len(data_test)
myNet.cuda()

#梯度优化以及loss
optimizer = optim.SGD(myNet.parameters(), lr=config.lr, weight_decay=config.weight_decay, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()

#学习率变化
def adjust_learning_rate(epoch):
    lr = config.lr

    if epoch > 650:
        lr = lr / 1000000
    elif epoch > 500:
        lr = lr / 100000
    elif epoch > 320:
        lr = lr / 10000
    elif epoch > 120:
        lr = lr / 1000
    elif epoch > 80:
        lr = lr / 100
    elif epoch > 30:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

# def test():
#     myNet.eval()
#     test_acc = 0.0
#     total = 0.0
#     for i, (images, labels) in enumerate(test_loader):
#         images = images.cuda()
#         labels = labels.cuda()
#         # Predict classes using images from the test set
#         outputs = myNet(images)
#         _, prediction = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         test_acc += (labels == prediction).sum().item()
#         # test_acc += torch.sum(prediction == labels.data)
#     # Compute the average acc and loss over all 10000 test images
#     test_acc = test_acc / total
#     return test_acc

max_batch = train_len // config.batch_size
def train():
    best_acc = 0.0
    for epoch in range(config.epochs):
        print("Epoch {}".format(epoch))
        print('I am training, please wait...')
        myNet.train()
        batch = 0
        train_acc = 0.0
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # 若GPU可用，将图像和标签移往GPU
            images = images.cuda()
            labels = labels.cuda()
            # 清除所有累积梯度
            optimizer.zero_grad()
            # 用来自测试集的图像预测类
            outputs = myNet(images)
            # 根据实际标签和预测值计算损失
            loss = loss_fn(outputs, labels)
            batch_loss = loss.cpu().item() * images.size(0)
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
            if batch % 1 == 0:
                print("Epoch: %d/%d || batch:%d/%d train_loss: %.3f || train_acc: %.2f"
                      % (epoch, config.epochs, batch, max_batch, batch_loss, batch_acc))
        # 调用学习率调整函数
        adjust_learning_rate(epoch)
        # 计算模型在50000张训练图像上的准确率和损失值
        train_acc = train_acc / train_len
        train_loss = train_loss / train_len
        # 用测试集评估
        # test_acc = test()
        # torch.save(myNet.state_dict(), config.model_save + "/" + "model_{}.pth".format(epoch))
        # 若测试准确率高于当前最高准确率，则保存模型
        if train_acc > best_acc:
            torch.save(myNet.state_dict(), config.model_save + "/" + "mask_better_{}.pth".format(epoch))
            best_acc = train_acc

        if (epoch % 5 == 0 and epoch > 0):
            torch.save(myNet.state_dict(), config.model_save + "/" + "mask_{}.pth".format(epoch))

        # 打印度量
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {}".format(epoch, train_acc, train_loss))

    torch.save(myNet.state_dict(), config.model_save + "/" + "mask_final_{}.pth".format(epoch))
# train()
if __name__ == '__main__':
    train()