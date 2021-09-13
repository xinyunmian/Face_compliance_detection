import numpy as np

class beauty_config(object):
    #1.string parameters
    train_data = "D:/data/imgs/makeup/crop_face"
    train_txt = "D:/data/imgs/makeup/train.txt"
    model_save = "D:/codes/pytorch_projects/face_makeup/BeautyDetect/weights"

    #2.numeric parameters
    epochs = 251
    batch_size = 16
    img_height = 128
    img_width = 128
    lr = 0.01
    weight_decay = 0.0005
    resetNet = "D:/codes/pytorch_projects/face_makeup/BeautyDetect/weights/beauty.pth"

config = beauty_config()
