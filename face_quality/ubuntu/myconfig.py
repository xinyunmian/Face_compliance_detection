import numpy as np

class Quality_config(object):
    #1.string parameters
    train_data = "/home/batman/WorkSpace/xym/images/face_quality"
    train_txt = "/home/batman/WorkSpace/xym/images/blur_shuffle.txt"
    val_data = "/home/batman/WorkSpace/xym/images/face_quality"
    val_txt = "/home/batman/WorkSpace/xym/images/blur_shuffle.txt"
    model_save = "/home/batman/WorkSpace/xym/face_quality/weights"

    #2.numeric parameters
    epochs = 501
    batch_size = 32
    crop_size = 96
    crop_num_scale = 15
    lr = 0.001
    weight_decay = 0.0005

    net_channels = [16, 32, 64, 96, 128]
    lad_channel = 32
    slim_channels = [16, 32, 64, 128, 256]

    resetNet = "/home/batman/WorkSpace/xym/face_quality/weights/pretrain.pth"

config = Quality_config()
