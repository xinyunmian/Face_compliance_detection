import numpy as np
import os
import cv2
import random
import shutil
import xml.etree.ElementTree as ET

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

def create_FaceQuality_label(imgdirs, txtsave):
    label_classfication = open(txtsave, mode="w+")
    for root, dirs, files in os.walk(imgdirs):
        for file in files:
            rootsplit = root.replace('\\', '/').split("/")
            dir = rootsplit[-1]
            imgpath = dir + "/" + file
            splitfile = file.split(".")[0]
            namesplit = splitfile.split("_")
            savedata = imgpath + " " + namesplit[-1]
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

def str_in_lists(strr, wlist=["0", "1"]):
    if strr[-3] in wlist and strr[-2] in wlist and strr[-1] in wlist:
        return 1
    else:
        return 0

def classificy_imgs_into_dir(imgdir, savedir):
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            imgpath = imgdir + "/" + file
            splitfile = file.split(".")[0]
            namesplit = splitfile.split("_")
            gender = int(namesplit[-3])
            obscure = int(namesplit[-2])
            bright = int(namesplit[-1])
            in_or_not = str_in_lists(strr=namesplit)
            if in_or_not == 1:
                if obscure == 1 or bright == 1:
                    if gender == 1:
                        savepath = savedir + "/boy/" + file
                        shutil.move(imgpath, savepath)
                    else:
                        savepath = savedir + "/girl/" + file
                        shutil.move(imgpath, savepath)

if __name__ == "__main__":
    imgpath = "D:/data/imgs/facePicture/blur/faces"
    txtpath = "D:/data/imgs/facePicture/blur/blur1.txt"
    shufflepath = "D:/data/imgs/facePicture/blur/blur1_shuffle.txt"
    # create_FaceQuality_label(imgpath, txtpath)
    # shuffle_txt(txtpath, shufflepath)

    dirp = "D:/wx/1117"
    dirs = "D:/wx_select/obscure_bright"
    classificy_imgs_into_dir(dirp, dirs)
    # img_augment(dirp, dirs)

















