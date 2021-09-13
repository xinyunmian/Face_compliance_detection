import numpy as np

class test_conf(object):
    model_save = "D:/codes/mask_detect/train/weights"
    mask_model = "better_mask_255.pth"

    # retinaface conf
    detect_size = 300
    origin_size = False
    confidence_thresh = 0.65
    nms_thresh = 0.35
    rfb_model = "face.pth"
    rfb = {
        'name': 'RFB',
        'min_sizes': [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]],
        'steps': [8, 16, 32, 64],
        'variance': [0.1, 0.2],
        'clip': False,
        'loc_weight': 2.0,
        'gpu_train': True,
        'batch_size': 32,
        'ngpu': 1,
        'epoch': 250,
        'decay1': 80,
        'decay2': 150,
        'decay3': 200,
        'image_size': 300
    }

    #2.numeric parameters
    epochs = 150
    batch_size = 32
    img_height = 128
    img_width = 128
    num_classes = 2
    lr = 0.01
    weight_decay = 0.0005
    bgr_mean = np.array([104, 117, 123], dtype=np.float32)
    bgr_std = np.array([58, 57, 59], dtype=np.float32)

config = test_conf()
