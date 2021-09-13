import numpy as np

class AgeGender_config(object):
    #1.string parameters
    train_data = "D:/data/imgs/facePicture/Age_Gender/align"
    train_txt = "D:/data/imgs/facePicture/Age_Gender/AgeGender_shuffle.txt"
    val_data = "/opt_ext_one/Documents/xym/imgdata"
    val_txt = "/opt_ext_one/Documents/xym/imgdata/test.txt"
    model_save = "D:/codes/project/age_gender/weights"

    #2.numeric parameters
    epochs = 150
    batch_size = 8
    img_height = 128
    img_width = 128
    lr = 0.01
    weight_decay = 0.0005
    bgr_mean = np.array([104, 117, 123], dtype=np.float32)
    bgr_std = np.array([58, 57, 59], dtype=np.float32)

    resetNet = "/opt_ext_one/Documents/xym/train_mask/weights/pretrain.pth"

config = AgeGender_config()
