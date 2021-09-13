import cv2
import os
import random

def change_one(a=1):
    img = cv2.imread('D:/data/imgs/facePicture/face_bright/face_skin/youwu/16.jpg')
    cv2.imshow('original_img', img)
    rows, cols, channels = img.shape
    dst = img.copy()
    b = 0
    for i in range(rows):
        for j in range(cols):
            for c in range(3):
                color = img[i, j][c] * a + b
                dst[i, j][c] = color
                if color > 255:  # 防止像素值越界（0~255）
                    dst[i, j][c] = 255
                elif color < 0:  # 防止像素值越界（0~255）
                    dst[i, j][c] = 0

    cv2.imshow('dst', dst)
    cv2.waitKey(0)

def change_light(imgpath, a):
    img = cv2.imread(imgpath)
    rows, cols, channels = img.shape
    dst = img.copy()
    for i in range(rows):
        for j in range(cols):
            for c in range(3):
                color = img[i, j][c] * a + 0
                dst[i, j][c] = color
                if color > 255:  # 防止像素值越界（0~255）
                    dst[i, j][c] = 255
                elif color < 0:  # 防止像素值越界（0~255）
                    dst[i, j][c] = 0
    return dst

def change_light_dir(imgdir, savedir):
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            splitfile = file.split("_b")
            imgname = splitfile[0]
            hz = file.split(".")[1]
            imgpath = root + "/" + file
            ratio = random.choice([0.9, 0.8, 0.7, 1.1, 1.26, 1.38])
            result = change_light(imgpath, ratio)
            if ratio == 0.9 or ratio == 1.1:
                savename = imgname + "_b_-1_d_-1_u_-1_c_-1_r_1." + hz
            if ratio == 0.8 or ratio == 1.26:
                savename = imgname + "_b_-1_d_-1_u_-1_c_-1_r_2." + hz
            if ratio == 0.7 or ratio == 1.38:
                savename = imgname + "_b_-1_d_-1_u_-1_c_-1_r_3." + hz
            savepath = savedir + "/" + savename
            cv2.imwrite(savepath, result)

if __name__ == '__main__':
    imgd = "D:/data/imgs/facePicture/face_bright/face_skin/standard_images/change"
    saved = "D:/data/imgs/facePicture/face_bright/face_skin/standard_images/contrast2"
    change_light_dir(imgd, saved)
    # change_one(a=1.3)





