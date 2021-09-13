class face_classification_config(object):
    #1.string parameters
    train_data = "D:/huoti_detect/imgdata/train"
    train_txt = "D:/huoti_detect/imgdata/train/train_shuffle.txt"
    test_data = ""
    val_data = "D:/huoti_detect/imgdata/train"
    val_txt = "D:/huoti_detect/imgdata/train/test.txt"
    model_save = "D:/huoti_detect/codes/train_face_spoofing/weights/20200314"
    gpus = "1"

    #2.numeric parameters
    epochs = 151
    batch_size = 64
    img_height = 128
    img_width = 128
    num_classes = 2
    seed = 888
    lr = 0.001
    lr_decay = 1e-4
    weight_decay = 0.0001
    rgb_mean = (104, 117, 123)

    resetNet = "D:/huoti_detect/codes/train_face_spoofing/weights/20200314/huoti_35.pth"

config = face_classification_config()
