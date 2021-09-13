import numpy as np

class train_yolo_config(object):
    #2.numeric parameters
    epochs = 501
    batch_size = 32
    netw = 320
    neth = 320
    boxes_maxnum = 50
    classes = 6
    nanchors = 5
    anchors = [0.68,1.38,0.97,1.87,1.03,2.65,1.31,2.68,1.63,3.12]
    # 1.25, 0.45, 2.8, 1.5, 3.75, 1.75, 5.9, 2.0, 6.8, 2.75 mark,ID
    # 0.31, 0.62, 0.45, 0.95, 0.55, 1.41, 0.62, 1.09, 0.78, 1.56 224*224 ears

    # data augment
    data_list = "D:/data/imgs/facePicture/ears/ears_shuffle.txt"
    letter_box = 0
    flip = 0
    blur = 0
    gaussian = 0
    saturation = 1.5
    exposure = 1.5
    hue = .1
    jitter = 0.2
    mixup = 3
    moasic = 0

    model_save = "D:/codes/pytorch_projects/ears_right/weights"
    lr = 0.001
    weight_decay = 0.0005
    bgr_mean = np.array([104, 117, 123], dtype=np.float32)
    bgr_std = np.array([58, 57, 59], dtype=np.float32)

    # test
    conf_thresh = 0.5
    nms_thresh = 0.3

traincfg = train_yolo_config()