import numpy as np

class Pose_config(object):
    #1.string parameters
    train_data = "D:/data/imgs/facePicture/pose_person/pose_shouder/head"
    train_txt = "D:/data/imgs/facePicture/pose_person/pose_shouder/pose_shuffle.txt"
    model_save = "D:/codes/pytorch_projects/person_pose/weights"

    #2.numeric parameters
    epochs = 501
    batch_size = 8
    img_height = 256
    img_width = 256
    lr = 0.01
    weight_decay = 0.0005

    resetNet = "D:/codes/pytorch_projects/person_pose/weights/pretrain.pth"

config = Pose_config()
