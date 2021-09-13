import numpy as np

class face_classification_config(object):
    #1.string parameters
    train_data = "/opt_ext_one/Document/xym/mouth_mask/data"
    train_txt = "/opt_ext_one/Document/xym/mouth_mask/data/train.txt"
    val_data = "/opt_ext_one/Document/xym/mouth_mask/data"
    val_txt = "/opt_ext_one/Document/xym/mouth_mask/data/train.txt"

    model_save = "/opt_ext_one/Document/xym/mouth_mask/codes/weights"

    #2.numeric parameters
    epochs = 150
    batch_size = 32
    img_height = 128
    img_width = 128
    num_classes = 2
    lr = 0.01
    weight_decay = 0.0005
    rgb_mean = (104, 117, 123)
    bgr_mean = np.array([104, 117, 123], dtype=np.float32)
    bgr_std = np.array([58, 57, 59], dtype=np.float32)

    resetNet = "D:/huoti_detect/codes/train_face_spoofing/weights/huoti_27.pth"

config = face_classification_config()
