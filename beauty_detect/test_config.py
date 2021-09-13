import numpy as np

class test_conf(object):
    model_save = "D:/codes/mask_detect/train/weights"
    mask_model = "better_mask_255.pth"

    # retinaface conf
    detect_size = 300
    origin_size = False
    confidence_thresh = 0.65
    nms_thresh = 0.35
    _model = "detect.pth"

    #2.numeric parameters
    epochs = 150
    batch_size = 32
    img_height = 128
    img_width = 128
    num_classes = 2
    lr = 0.01
    weight_decay = 0.0005

configt = test_conf()
