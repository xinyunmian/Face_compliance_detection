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

hist1 = "D:/data/imgs/facePicture/shadow/getmask/yy.txt"
hist2 = "D:/data/imgs/facePicture/shadow/getmask/nm.txt"

corr1 = np.loadtxt(hist1)
corr2 = np.loadtxt(hist2)
# corr1 = np.sort(corr1)
# corr2 = np.sort(corr2)
len1 = len(corr1)
len2 = len(corr2)

errx = np.arange(len1)
nmx = np.arange(len2)
plt.plot(errx, corr1, 'r', label='error')
plt.plot(nmx, corr2, 'g.-', label='norm')

# eexp = normal_distribution(corr1)
# rexp = normal_distribution(corr2)
# plt.plot(corr1, eexp, 'r', label='error')
# plt.plot(corr2, rexp, 'g.-', label='norm')



# yright = 2 * np.ones([254,])
# yerror = np.ones([254,])
#
# plt.plot(corr1, yerror, '.-', label='错误')
# plt.plot(corr2, yright, '.-', label='正确')

# xkd = np.arange(0, 1, 0.05)
# plt.xticks(xkd)  # 设置横坐标刻度
ykd = np.arange(0, 1.5, 0.05)
plt.yticks(ykd)  # 设置纵坐标刻度
plt.xlabel('相关性') # 设置横坐标轴标题
plt.legend() # 显示图例，即每条线对应 label 中的内容
plt.show() # 显示图形

print("done")













