import numpy as np

class face_classification_config(object):
    #1.string parameters
    train_data = "D:/data/imgs/facePicture/glasses/train"
    train_txt = "D:/data/imgs/facePicture/glasses/shuffle_train.txt"
    val_data = "D:/data/imgs/facePicture/glasses/train"
    val_txt = "D:/data/imgs/facePicture/glasses/shuffle_train.txt"

    model_save = "D:/codes/glasses_detect/weights"

    #2.numeric parameters
    epochs = 301
    batch_size = 8
    img_height = 64
    img_width = 128
    num_classes = 4
    lr = 0.01
    weight_decay = 0.0005
    bgr_mean = np.array([104, 117, 123], dtype=np.float32)
    bgr_std = np.array([58, 57, 59], dtype=np.float32)

    resetNet = "D:/codes/glasses_detect/weights/glasses.pth"

config = face_classification_config()
