cfg_mnet = {
    # 'min_sizes': [[15, 32, 48], [64, 96], [128, 192], [256, 384, 512]],#mobilev3_small_148
    # 'min_sizes': [[8, 15, 24], [40, 64], [96, 162], [256, 480, 576]],#mobilev3_small_102,mobilev3_small_98, mobilev3_small_248
    # 'min_sizes': [[10, 20, 30], [48, 64], [96, 128], [192, 256, 512]],#Facev3_small_104,Facev3_small_244
    'min_sizes': [[15, 30], [48, 96], [128, 192], [256, 480]],
    # 'min_sizes': [[16, 32], [48, 96], [128, 224], [384, 576]],
    # 'min_sizes': [[10, 20], [40, 80], [120, 200], [256, 480]],
    'steps': [8, 16, 32, 64],
    'variance': [0.1, 0.2],
    'loc_weight': 2.0,
    'clip': False,
    'batch_size': 8,
    'epoch': 250,
    'decay1': 100,
    'decay2': 200,
    'image_size': 640,
    'rgb_mean': (104, 117, 123),
    'std_mean': (58, 57, 59),
    'out_channels': [16, 24, 32, 64, 96, 128],
    'fpn_in_list': [32, 64, 96, 128],
    'fpn_out_list': [32, 64, 96, 128],
    'ssh_out_channel': 64,
    'num_classes': 2
}

cfg_slim = {
    'min_sizes': [[10, 20], [40, 80], [120, 200], [256, 480]],#mobilev3_150
    # 'min_sizes': [[10, 20, 30], [48, 64], [96, 128], [192, 256, 512]],
    'steps': [8, 16, 32, 64],
    'variance': [0.1, 0.2],
    'loc_weight': 2.0,
    'clip': False,
    'batch_size': 8,
    'epoch': 250,
    'decay1': 100,
    'decay2': 200,
    'image_size': 640,
    'rgb_mean': (104, 117, 123),
    'std_mean': (58, 57, 59),
    'num_classes': 2
}

cfg_mn3 = {
    'min_sizes': [[5, 10, 20, 36], [48, 64, 96, 128], [192, 256, 384, 576]],
    'steps': [16, 32, 64],
    'variance': [0.1, 0.2],
    'loc_weight': 2.0,
    'clip': False,
    'batch_size': 8,
    'epoch': 250,
    'decay1': 100,
    'decay2': 200,
    'image_size': 640,
    'rgb_mean': (104, 117, 123),
    'std_mean': (58, 57, 59),
    'num_classes': 2
}