import numpy as np
import os
import cv2
import random
import shutil

def get_img_list(imgdir, listpath):
    list_file = open(listpath, 'w+')
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            if file.endswith("txt"):
                continue
            else:
                root = root.replace('\\', '/')
                imgpath = root + "/" + file
                list_file.write(imgpath + "\n")
    list_file.close()

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


if __name__ == "__main__":
    imgd = "D:/data/imgs/ornament_IDphoto/train"
    txtlist = "D:/codes/yolov4/darknet/build/darknet/x64/yolo4_tiny/train.txt"
    get_img_list(imgd, txtlist)

    txtpath2 = "D:/data/imgs/facePicture/ears/ears.txt"
    shufflepath = "D:/codes/yolov4/darknet/build/darknet/x64/yolo4_tiny/shuffe_train.txt"
    shuffle_txt(txtlist, shufflepath)


















