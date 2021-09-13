import numpy as np

class faceBright_config(object):
    #1.string parameters
    train_data = "D:/data/imgs/facePicture/face_bright/face_skin/standard_images"
    train_txt = "D:/data/imgs/facePicture/face_bright/faceskin.txt"
    val_data = "D:/data/imgs/facePicture/face_bright/face_skin/standard_images"
    val_txt = "D:/data/imgs/facePicture/face_bright/faceskin.txt"
    model_save = "D:/codes/pytorch_projects/faceBright_detect/weights"

    #2.numeric parameters
    epochs = 150
    batch_size = 4
    img_height = 128
    img_width = 128
    lr = 0.01
    weight_decay = 0.0005
    bgr_mean = np.array([104, 117, 123], dtype=np.float32)
    bgr_std = np.array([58, 57, 59], dtype=np.float32)

    resetNet = "D:/codes/pytorch_projects/faceBright_detect/weights/pretrain.pth"

config = faceBright_config()
