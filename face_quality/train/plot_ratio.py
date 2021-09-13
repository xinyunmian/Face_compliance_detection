import os
import random
import math

import cv2
import numpy as np
import matplotlib.pyplot as plt

def normal_distribution(x, mean=0.4, sigma=0.05):
    return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi) * sigma)

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

hist1 = "D:/data/imgs/facePicture/blur/test/result_1.txt"
hist2 = "D:/data/imgs/facePicture/blur/test/result_2.txt"
label = "D:/data/imgs/facePicture/blur/test/label.txt"

corr1 = np.loadtxt(hist1)
corr2 = np.loadtxt(hist2)
corr_lab = np.loadtxt(label)
wucha1 = np.mean(abs(corr1-corr_lab))
wucha2 = np.mean(abs(corr2-corr_lab))


index = np.argsort(corr_lab)
corr1 = corr1[index]
corr2 = corr2[index]
corr_lab = corr_lab[index]
len_lab = len(corr_lab)

labx = np.arange(len_lab)
plt.plot(labx, corr_lab, 'ro-', label='实际值')
plt.plot(labx, corr1, 'b.-', label='预测值-传统')
# plt.plot(labx, abs(corr1-corr_lab), 'yo-', label='误差-传统')
plt.plot(labx, corr2, 'g.-', label='预测值-改进')
# plt.plot(labx, abs(corr2-corr_lab), 'c*-', label='误差-改进')

# eexp = normal_distribution(corr1)
# rexp = normal_distribution(corr2)
# plt.plot(corr1, eexp, 'r', label='error')
# plt.plot(corr2, rexp, 'g.-', label='norm')



# yright = 2 * np.ones([254,])
# yerror = np.ones([254,])
#
# plt.plot(corr1, yerror, '.-', label='错误')
# plt.plot(corr2, yright, '.-', label='正确')

xkd = np.arange(0, 310, 30)
plt.xticks(xkd)  # 设置横坐标刻度
ykd = np.arange(0, 1.1, 0.1)
plt.yticks(ykd)  # 设置纵坐标刻度
plt.xlabel('样本个数')
plt.ylabel('预测值与实际值的差') # 设置横坐标轴标题
plt.text(25, .80, "测试样本数300:误差如下", color="b")
plt.text(25, .73, "总误差(传统:14.7，改进:10.5)", color="g")
plt.text(25, .66, "均值误差(传统:0.049，改进:0.035)", color="r")
plt.legend(loc="lower right") # 显示图例，即每条线对应 label 中的内容
plt.show() # 显示图形

print("done")













