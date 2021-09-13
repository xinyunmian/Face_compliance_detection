import numpy as np

class AgeGender_config(object):
    #1.string parameters
    train_data = "/opt_ext_one/Document/xym/AgeGender/face_align"
    train_txt = "/opt_ext_one/Document/xym/AgeGender/AgeGender_shuffle.txt"
    val_data = "/opt_ext_one/Documents/xym/imgdata"
    val_txt = "/opt_ext_one/Documents/xym/imgdata/test.txt"
    model_save = "/opt_ext_one/Document/xym/train_AGNet/weights"

    #2.numeric parameters
    epochs = 301
    batch_size = 64
    img_height = 128
    img_width = 128
    lr = 0.01
    weight_decay = 0.0005
    bgr_mean = np.array([104, 117, 123], dtype=np.float32)
    bgr_std = np.array([58, 57, 59], dtype=np.float32)

    resume = True
    resetNet = "/opt_ext_one/Document/xym/train_AGNet/weights/Agpretrain.pth"

config = AgeGender_config()
