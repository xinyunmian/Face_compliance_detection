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

def create_pose_label(imgdirs, txtsave):
    label_classfication = open(txtsave, mode="w+")
    for root, dirs, files in os.walk(imgdirs):
        for file in files:
            rootsplit = root.replace('\\', '/').split("/")
            dir = rootsplit[-1]
            imgpath = dir + "/" + file
            split_file = file.split("_")
            savedata = imgpath + " " + split_file[0] + " " + split_file[1] + " " + split_file[2]
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

def add_ClassLabel(imgdir, addclass="1"):
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            file_split = file.split("_")
            root = root.replace('\\', '/')
            imgpath = root + "/" + file
            # replace_path = root + "/" + addclass + "_" + file
            # replace_path = root + "/" + file_split[0] + "_" + addclass + "_" + file_split[2]
            replace_path = root + "/" + addclass + "_" + file_split[1] + "_" + file_split[2]
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

def resize_dir(imgdirs, savedirs):
    for root, dirs, files in os.walk(imgdirs):
        for file in files:
            rootsplit = root.replace('\\', '/').split("/")
            dir = rootsplit[-1]
            imgpath = imgdirs + "/" + dir + "/" + file
            savepath = savedirs + "/" + dir + "/" + file
            img = cv2.imread(imgpath)

            interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
            interp_method = interp_methods[random.randrange(5)]
            img = cv2.resize(img, (256, 256), interpolation=interp_method)

            cv2.imwrite(savepath, img)

def rename_dir(imgdir):
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            # root = root.replace('\\', '/')
            filesplit = file.split("_")
            filename = filesplit[-1]
            if len(filesplit) > 4:
                filename = filesplit[-2] + filesplit[-1]
            imgpath = imgdir + "/" + file
            replace_path = imgdir +  "/" + filename  # file_nospace = file.replace(" - 副本", "")
            # replace_path = imgdir + "/3_" + file
            os.rename(imgpath, replace_path)

def delete_img(errdir, imgdir):
    for root, dirs, files in os.walk(errdir):
        for file in files:
            imgpath = imgdir + "/" + file
            img = cv2.imread(imgpath)
            if img is not None:
                os.remove(imgpath)

if __name__ == "__main__":
    imgpath = "D:/data/imgs/facePicture/pose_person/pose_shouder/train/copsNormal"
    txtpath = "D:/data/imgs/facePicture/pose_person/pose_shouder/pose1.txt"
    shufflepath = "D:/data/imgs/facePicture/pose_person/pose_shouder/pose1_shuffle.txt"
    create_pose_label(imgpath, txtpath)
    shuffle_txt(txtpath, shufflepath)

    # remove_space(imgpath)
    # add_ClassLabel(imgpath, addclass="2")

    dirp = "D:/data/imgs/facePicture/pose_person/pose_shouder/train/my2"
    dirs = "D:/data/imgs/facePicture/pose_person/pose_shouder/zy/1"
    # rename_dir(dirp)
    # delete_img(dirp, dirs)
    # img_augment(dirp, dirs)
    # rotate_dir(dirp, dirs)
    # resize_dir(dirp, dirs)

















