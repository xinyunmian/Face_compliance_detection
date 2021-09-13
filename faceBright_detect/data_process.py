import numpy as np
import os
import cv2
import random
import shutil
import xml.etree.ElementTree as ET

def create_age_gender_label(imgdir, txtsave):
    age_gender = open(txtsave, mode = "w+")
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            if file.endswith("jpg"):
                root = root.replace('\\', '/')
                imgpath = root + "/" + file
                dir, imgname = os.path.split(imgpath)
                splitname = file.split("_")#获取年龄，性别
                tage = splitname[0]#年龄
                tgender = splitname[1]#性别  0:男  1:女
                splitdir = dir.split("/")#获取文件夹名称
                subdir_name = splitdir[-1]

                savep = subdir_name + "/" + imgname
                savedata = savep + " " + tage + " " + tgender#utkface_align1/a.jpg 100 1
                age_gender.write(savedata)
                age_gender.write("\n")
    age_gender.close()

def create_age_gender_label2(imgdir, txtsave):
    age_gender = open(txtsave, mode = "w+")
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            if file.endswith("jpg"):
                root = root.replace('\\', '/')
                splitroot = root.split("/")
                imgpath = root + "/" + file
                dir, imgname = os.path.split(imgpath)
                splitname = file.split("A")#获取年龄，性别
                splitname2 = splitname[1].split(".")  # 获取年龄
                tage = splitname2[0].strip('r')#年龄
                tgender = splitroot[-1]#性别  0:男  1:女
                splitdir = dir.split("/")#获取文件夹名称
                subdir_name = splitdir[-1]

                savep = "all_faces2/" + imgname
                savedata = savep + " " + tage + " " + tgender#utkface_align1/a.jpg 100 1
                age_gender.write(savedata)
                age_gender.write("\n")
    age_gender.close()

def create_classfication_label(imgdir, txtsave):
    label_classfication = open(txtsave, mode="w+")
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            if file.endswith("jpg"):
                root = root.replace('\\', '/')
                splitroot = root.split("/")
                dirname = splitroot[-1]
                imgpath = dirname + "/" + file
                savedata = imgpath + " " + dirname
                label_classfication.write(savedata)
                label_classfication.write("\n")
    label_classfication.close()

def img_augment(imgdir, savedir):
    imgid = 0
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            root = root.replace('\\', '/')
            imgname, houzui = file.split(".")
            imgpath = root + "/" + file
            img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
            # imgmirror = cv2.flip(img, 1)
            savepath1 = savedir + "/normal" + str(imgid) + "_0." + houzui
            savepath2 = savedir + "/mshadow" + str(imgid) + "_1." + houzui
            cv2.imwrite(savepath1, img)
            # cv2.imwrite(savepath2, imgmirror)
            imgid += 1

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

#文件夹1有若干图片，文件夹2有很多图片，移除和文件夹1相同的文件
def remove_filesjpg(imgdir, dirremove, save):
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            if file.endswith("jpg"):
                filelist = list(file)
                filelist.pop(-5)
                filep = "".join(filelist)

                remove_name = dirremove + "/" + filep
                save_name = save + "/" + filep
                img = cv2.imread(remove_name, cv2.IMREAD_COLOR)
                if img is not None:
                    shutil.copy(remove_name, save_name)

def select_txt_by_img(imgpath, txtdir, save):
    f1 = open(imgpath, 'r+')
    lines = f1.readlines()
    for i in range(len(lines)):
        line = lines[i].strip("\n")
        img_name = line.split("/")[-1]
        txtname = img_name.replace("jpg", "txt")
        savetxt = save + "/" + txtname

        txtpath = txtdir + "/" + txtname
        shutil.move(txtpath, savetxt)


#txt中有若干图片路径，将图片移动到另一个文件夹
def remove_filestxt(txtpath, dir, save):
    f1 = open(txtpath, 'r+')
    lines = f1.readlines()
    for i in range(len(lines)):
        line = lines[i].strip("\n")
        img_name = line.split("/")[-1]
        imgpath = dir + "/" + img_name
        txtpath = imgpath.replace("jpg", "txt")
        saveimg = save + "/" + img_name
        savetxt = saveimg.replace("jpg", "txt")

        img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        if img is not None:
            shutil.move(imgpath, saveimg)
            shutil.move(txtpath, savetxt)

def get_img_list(imgdir, listpath, endname):
    list_file = open(listpath, 'w+')
    i = 0
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            i += 1
            if file.endswith(endname):
                root = root.replace('\\', '/')
                imgpath = root + "/" + file
                list_file.write(imgpath + "\n")
    list_file.close()

def create_faceBright_label(imgdir, txtsave):
    label_classfication = open(txtsave, mode="w+")
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            if file.endswith("jpg"):
                splitfile = file.split(".jpg")
                imgname = splitfile[0]
                namesplit = imgname.split("_")
                root_split = root.split("\\")
                imgpath = root_split[1] + "/" + file
                savedata = imgpath + " " + namesplit[-3] + " " + namesplit[-2] + " " + namesplit[-1]
                label_classfication.write(savedata)
                label_classfication.write("\n")
    label_classfication.close()

def create_FaceSkin_label(imgdirs, txtsave):
    label_classfication = open(txtsave, mode="w+")
    for root, dirs, files in os.walk(imgdirs):
        for file in files:
            rootsplit = root.replace('\\', '/').split("/")
            dir = rootsplit[-1]
            imgpath = dir + "/" + file
            splitfile = file.split(".")[0]
            namesplit = splitfile.split("_")

            # splitfile = file.split(".")[0]
            # namesplit = splitfile.split("_")
            # root_split = root.split("/")
            # imgpath = root_split[-1] + "/" + file
            # bright dark yinyang skin ratio
            savedata = imgpath + " " + namesplit[-9] + " " + namesplit[-7] + " " + namesplit[-5] + " " + namesplit[-3] + " " + namesplit[-1]
            label_classfication.write(savedata)
            label_classfication.write("\n")
    label_classfication.close()

def get_faceBright_normal_images(imgdir, savedir):
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            if file.endswith("jpg"):
                splitfile = file.split(".jpg")
                imgname = splitfile[0]
                namesplit = imgname.split("_")
                dark = int(namesplit[-3])
                bright = int(namesplit[-2])
                yin_yang = int(namesplit[-1])
                he = dark + bright + yin_yang
                if he == 0:
                    root = root.replace("\\", "/")
                    srcpath = root + "/" + file
                    dstpath = savedir + "/" + file
                    shutil.move(srcpath, dstpath)


if __name__ == "__main__":
    imgpath = "D:/data/imgs/facePicture/shadow/select_normal"
    savepath = "D:/data/imgs/facePicture/shadow/normal1"
    txtpath = "D:/data/imgs/facePicture/face_bright/faceskin_crop.txt"
    # create_age_gender_label(imgpath, txtpath)
    # create_age_gender_label2(imgpath, txtpath)
    # create_classfication_label(imgpath, txtpath)
    # create_faceBright_label(imgpath, txtpath)
    # create_FaceSkin_label(imgpath, txtpath)
    # get_faceBright_normal_images(imgpath, savepath)

    txtpath2 = "D:/data/imgs/facePicture/mask/mouth_mask/mouth_mask.txt"
    shufflepath = "D:/data/imgs/facePicture/mask/mouth_mask/shuffle_mask.txt"
    # shuffle_txt(txtpath2, shufflepath)

    imgd = "D:/data/imgs/facePicture/face_bright/train/aa"
    txtlist = "D:/data/imgs/rename/watermask_ID/mask_ID.txt"
    # get_img_list(imgd, txtlist, endname="jpg")

    img_augment(imgpath, savepath)

















