import numpy as np
import os
import cv2
import random
import shutil

def shuffle_txt(srctxt, shuffletxt):
    FileNamelist = []
    files = open(srctxt, 'r+')
    for line in files:
        line = line.strip('\n')  # 删除每一行的\n
        FileNamelist.append(line)
    print('len ( FileNamelist ) = ', len(FileNamelist))
    files.close()
    random.shuffle(FileNamelist)

    file_handle = open(shuffletxt, mode='w+')
    for idx in range(len(FileNamelist)):
        str = FileNamelist[idx]
        file_handle.write(str)
        file_handle.write("\n")
    file_handle.close()

def create_NeckShadow_label(imgdirs, txtsave):
    label_classfication = open(txtsave, mode="w+")
    for root, dirs, files in os.walk(imgdirs):
        for file in files:
            rootsplit = root.replace('\\', '/').split("/")
            dir = rootsplit[-1]
            imgpath = dir + "/" + file
            splitfile = file.split(".")[0]
            namesplit = splitfile.split("_")
            savedata = imgpath + " " + namesplit[0]
            label_classfication.write(savedata)
            label_classfication.write("\n")
    label_classfication.close()

def img_augment(imgdir, savedir):
    imgid = 750
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            root = root.replace('\\', '/')
            imgname, houzui = file.split(".")
            imgpath = root + "/" + file
            img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
            imgmirror = cv2.flip(img, 1)
            savepath1 = savedir + "/shadow" + str(imgid) + "_1." + houzui
            savepath2 = savedir + "/mshadow" + str(imgid) + "_1." + houzui
            cv2.imwrite(savepath1, img)
            cv2.imwrite(savepath2, imgmirror)
            imgid += 1

def remove_space(imgdirs):
    for root, dirs, files in os.walk(imgdirs):
        for file in files:
            root = root.replace('\\', '/')
            file_nospace = file.replace(" ", "")
            imgpath = root + "/" + file
            replace_path = root + "/" + file_nospace
            os.rename(imgpath, replace_path)

def rotate_face(img, angle):
    imgh, imgw, imgc = img.shape
    center = (imgw / 2, imgh / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (imgw, imgh))
    return rotated

def rotate_dir(imgdir, savedir):
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            root = root.replace('\\', '/')
            imgpath = root + "/" + file
            savepath = savedir + "/" + file
            img = cv2.imread(imgpath)
            angle = random.choice([-30, -20, -10, 0, 10, 20, 30])
            rot_img = rotate_face(img, angle)
            cv2.imwrite(savepath, rot_img)

if __name__ == "__main__":
    imgpath = "D:/data/imgs/facePicture/shadow/train4"
    txtpath = "D:/data/imgs/facePicture/shadow/shadow.txt"

    txtpath2 = "D:/data/imgs/facePicture/shadow/shadow.txt"
    shufflepath = "D:/data/imgs/facePicture/shadow/shadow_shuffle.txt"
    # create_NeckShadow_label(imgpath, txtpath)
    # shuffle_txt(txtpath2, shufflepath)

    # remove_space(imgpath)

    dirp = "D:/data/imgs/facePicture/pose_person/img"
    dirs = "D:/data/imgs/facePicture/pose_person/save"
    # img_augment(dirp, dirs)
    rotate_dir(dirp, dirs)

















