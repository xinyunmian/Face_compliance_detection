import torch
import os
import torch.utils.data as data
from load_data import Age_GenderDataLoader, AG_DLDLDataLoader
from mobilev3_small import mobilev3_AgeGenderNet, mobilev3_AGDLDL
from myconfig import config
from Age_Gender_Loss import Age_GenderLoss, AGDLDL_Loss

from collections import OrderedDict
#cuda
torch.cuda.set_device(1)

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

#读取训练测试数据
data_train = AG_DLDLDataLoader(config.train_data, config.train_txt)
train_loader = data.DataLoader(data_train,batch_size=config.batch_size,shuffle=True,num_workers=0)
train_len = len(data_train)

#定义网络结构
AGNet = mobilev3_AGDLDL()
if config.resume:
    print('Loading resume network...')
    state_dict = torch.load(config.resetNet)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    AGNet.load_state_dict(new_state_dict)

AGNet.cuda()

criterion = AGDLDL_Loss()
optimizer = torch.optim.SGD([
                {'params': AGNet.parameters()},
                {'params': criterion.parameters(),}
            ], lr=config.lr, weight_decay=config.weight_decay, momentum=0.9)

def adjust_learning_rate(epoch):
    lr = config.lr
    if epoch > 650:
        lr = lr / 1000000
    elif epoch > 500:
        lr = lr / 100000
    elif epoch > 320:
        lr = lr / 10000
    elif epoch > 200:
        lr = lr / 1000
    elif epoch > 100:
        lr = lr / 100
    elif epoch > 40:
        lr = lr / 10
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

max_batch = train_len // config.batch_size

def train():
    for epoch in range(config.epochs):
        print("Epoch {}".format(epoch))
        print('I am training, please wait...')
        AGNet.train()
        batch = 0
        lr_cur = 0.01
        for i, (images, ages, genders) in enumerate(train_loader):
            batch += 1
            # 若GPU可用，将图像和标签移往GPU
            images = images.cuda()
            ages = ages.cuda()
            genders = genders.cuda()
            # 预测
            pre_age, pre_gender = AGNet(images)
            # 清除所有累积梯度
            optimizer.zero_grad()

            lossa, lossg, loss = criterion(pre_age, pre_gender, ages, genders)
            loss.backward()
            optimizer.step()

            for params in optimizer.param_groups:
                lr_cur = params['lr']
            print("Epoch:{}/{} || Epochiter: {}/{} || Age: {:.4f} Gender: {:.4f} Sum: {:.4f} || LR: {:.8f}"
                  .format(epoch, config.epochs, max_batch, batch, lossa.item(), lossg.item(), loss.item(), lr_cur))

            # 调用学习率调整函数
            adjust_learning_rate(epoch)
            torch.save(AGNet.state_dict(), config.model_save + "/" + "AgeGender_final.pth")

            if (epoch % 2 == 0 and epoch > 0):
                torch.save(AGNet.state_dict(), config.model_save + "/" + "AgeGender_{}.pth".format(epoch))

def trainDLDL():
    for epoch in range(config.epochs):
        print("Epoch {}".format(epoch))
        print('I am training, please wait...')
        AGNet.train()
        batch = 0
        lr_cur = 0.01
        for i, (images, ages, normages, genders) in enumerate(train_loader):
            batch += 1
            # 若GPU可用，将图像和标签移往GPU
            images = images.cuda()
            ages = ages.cuda()
            normages = normages.cuda()
            genders = genders.cuda()
            # 预测
            pre_age, pre_gender = AGNet(images)
            # 清除所有累积梯度
            optimizer.zero_grad()

            lossa, lossnorma, lossg, loss = criterion(pre_age, pre_gender, ages, normages, genders)
            loss.backward()
            optimizer.step()

            for params in optimizer.param_groups:
                lr_cur = params['lr']
            print("Epoch:{}/{} || Epochiter: {}/{} || Age: {:.4f} Gender: {:.4f} Sum: {:.4f} || LR: {:.8f}"
                  .format(epoch, config.epochs, max_batch, batch, lossa.item(), lossg.item(), loss.item(), lr_cur))

            # 调用学习率调整函数
            adjust_learning_rate(epoch)
            torch.save(AGNet.state_dict(), config.model_save + "/" + "AgeGender_final.pth")

            if (epoch % 2 == 0 and epoch > 0):
                torch.save(AGNet.state_dict(), config.model_save + "/" + "AgeGender_{}.pth".format(epoch))


if __name__ == "__main__":
    # train()
    trainDLDL()

















