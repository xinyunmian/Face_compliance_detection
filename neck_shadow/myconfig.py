import numpy as np

class Shadow_config(object):
    #1.string parameters
    train_data = "D:/data/imgs/facePicture/shadow/train"
    train_txt = "D:/data/imgs/facePicture/shadow/shadow_shuffle.txt"
    val_data = "D:/data/imgs/facePicture/face_bright/face_skin/standard_images"
    val_txt = "D:/data/imgs/facePicture/face_bright/faceskin.txt"
    model_save = "D:/codes/pytorch_projects/neck_shadow/weights"

    #2.numeric parameters
    epochs = 501
    batch_size = 32
    img_height = 96
    img_width = 128
    lr = 0.01
    weight_decay = 0.0005
    bgr_mean = np.array([104, 117, 123], dtype=np.float32)
    bgr_std = np.array([58, 57, 59], dtype=np.float32)

    resetNet = "D:/codes/pytorch_projects/faceBright_detect/weights/pretrain.pth"

config = Shadow_config()
