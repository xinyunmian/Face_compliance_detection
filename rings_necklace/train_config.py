import numpy as np

class train_yolo_config(object):
    #2.numeric parameters
    epochs = 1001
    batch_size = 8
    netw = 320
    neth = 320
    boxes_maxnum = 50
    label_class = 6
    classes = 6
    nanchors = 5
    anchors = [0.38,0.88, 0.94,2.2, 2.0,1.4, 3.0,3.5, 3.34,2.18]
    # 1.25, 0.45, 2.8, 1.5, 3.75, 1.75, 5.9, 2.0, 6.8, 2.75 mark,ID
    # 0.31, 0.62, 0.45, 0.95, 0.55, 1.41, 0.62, 1.09, 0.78, 1.56 224*224 ears

    out_channels = [16, 32, 64, 96, 128]
    target_outc = nanchors * (5 + label_class)

    # data augment
    data_list = "D:/codes/pytorch_projects/rings_lace_detect/rings_lace.txt"
    letter_box = 0
    flip = 1
    blur = 0
    gaussian = 0
    saturation = 1.5
    exposure = 1.5
    hue = .1
    jitter = 0.2
    mixup = 3
    moasic = 0

    model_save = "D:/codes/pytorch_projects/rings_necklace/weights"
    lr = 0.001
    weight_decay = 0.0005
    bgr_mean = np.array([104, 117, 123], dtype=np.float32)
    bgr_std = np.array([58, 57, 59], dtype=np.float32)

    # test
    conf_thresh = 0.5
    nms_thresh = 0.3

traincfg = train_yolo_config()