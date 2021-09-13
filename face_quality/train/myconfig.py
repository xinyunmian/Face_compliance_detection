import numpy as np

class Quality_config(object):
    #1.string parameters
    train_data = "D:/data/imgs/facePicture/blur/faces"
    train_txt = "D:/data/imgs/facePicture/blur/blur_shuffle.txt"
    val_data = "D:/data/imgs/facePicture/blur/faces"
    val_txt = "D:/data/imgs/facePicture/blur/blur_shuffle.txt"
    model_save = "D:/codes/pytorch_projects/face_quality/train/weights"

    epochs = 501
    batch_size = 8
    crop_size = 96
    crop_num = 25
    crop_scale = 15
    lr = 0.001
    weight_decay = 0.0005
    bgr_mean = np.array([104, 117, 123], dtype=np.float32)

    net_channels = [16, 32, 64, 96, 128]
    lad_channel = 32
    slim_channels = [16, 32, 64, 128, 256]

    resetNet = "D:/codes/pytorch_projects/faceBright_detect/weights/pretrain.pth"

config = Quality_config()
