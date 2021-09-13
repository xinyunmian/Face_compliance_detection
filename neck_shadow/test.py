import torch
import math
import numpy as np
import os
import cv2
import imutils
import random

from slim_net import NeckShadowNet
from myconfig import config as testconf
from load_data import pytorch_to_dpcoreParams, save_feature_channel

from detector.config import cfg_mnet as cfg
from detector.create_anchors import PriorBox
from detector.mobilev3_face import mobilev3Fpn_small
from detector.retinaface_utils import decode, decode_landm
from detector.nms import py_cpu_nms
device = "cpu"

def rand_ratio(id):
    rax1 = 0.0
    ray1 = 0.72
    rax2 = 1.0
    ray2 = 0.55
    if id == 2:
        rax1 = -0.05
        rax2 = 1.1
        ray2 = 0.65
    if id == 3:
        rax1 = 0.0
        rax2 = 1.0
        ray2 = 0.65
    if id == 4:
        rax1 = -0.05
        rax2 = 1.1
        ray2 = 0.75
    if id == 5:
        rax1 = -0.05
        rax2 = 1.1
        ray2 = 0.57
    if id == 6:
        rax1 = 0.0
        rax2 = 1.0
        ray2 = 0.75

    return rax1, ray1, rax2, ray2

def expand_facebox(rect, imgw, imgh):
    bx = rect[0]
    by = rect[1]
    bw = rect[2] - rect[0]
    bh = rect[3] - rect[1]

    #  face
    # nbx1 = bx - 0.1 * bw
    # nby1 = by - 0.1 * bh
    # nbx2 = nbx1 + 1.2 * bw
    # nby2 = nby1 + 1.1 * bh

    # neck
    # randid = random.choice([1, 2, 3, 4, 5, 6])
    # sx1, sy1, sx2, sy2 = rand_ratio(randid)
    sx1 = -0.03
    sy1 = 0.72
    sx2 = 1.06
    sy2 = 0.65

    nbx1 = bx + sx1 * bw
    nby1 = by + sy1 * bh
    nbx2 = nbx1 + sx2 * bw
    nby2 = nby1 + sy2 * bh

    pp = np.zeros(4, dtype=np.int32)
    rx1 = max(nbx1, 0)
    ry1 = max(nby1, 0)
    rx2 = min(nbx2, imgw)
    ry2 = min(nby2, imgh)

    pp[0] = rx1
    pp[1] = ry1
    pp[2] = rx2
    pp[3] = ry2
    return pp

def img_process(img):
    """将输入图片转换成网络需要的tensor
            Args:
                img_path: 人脸图片路径
            Returns:
                tensor： img(batch, channel, width, height)
    """
    im = cv2.resize(img, (testconf.img_width, testconf.img_height), interpolation=cv2.INTER_LINEAR)
    im = im.astype(np.float32)
    # im = (im - testconf.bgr_mean) / testconf.bgr_std
    im = im / 255.0
    im = im.transpose(2, 0, 1)
    im = torch.from_numpy(im)
    im = im.unsqueeze(0)
    im = im.to(device)
    return im

def detect_one_img(faceNet, img_data, minface):
    rgb_mean = cfg['rgb_mean']  # bgr order
    std_mean = cfg['std_mean']
    conf_thresh = 0.5
    nms_thresh = 0.35

    img = np.float32(img_data)
    im_shape = img.shape
    im_size_max = np.max(im_shape[0:2])
    # res_scal = 15 / float(minface)
    res_scal = 250 / im_size_max

    img = cv2.resize(img, None, None, fx=res_scal, fy=res_scal, interpolation=cv2.INTER_CUBIC)

    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    scale = scale.to(device)

    # 减去均值转成numpy
    im_height, im_width, _ = img.shape
    img -= rgb_mean
    img /= std_mean
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)

    loc, conf, landms = faceNet(img)  # forward pass
    pointb = torch.full_like(loc, 0)
    pointx = landms[:, :, [0, 2, 4, 6]]
    pointy = landms[:, :, [1, 3, 5, 7]]
    maxx, maxix = torch.max(pointx[0,:,:], 1)
    minx, minix = torch.min(pointx[0,:,:], 1)
    maxy, maxiy = torch.max(pointy[0,:,:], 1)
    miny, miniy = torch.min(pointy[0,:,:], 1)
    boxw = maxx - minx
    boxh = maxy - miny
    pointb[:, :, 0] = minx
    pointb[:, :, 1] = miny
    pointb[:, :, 2] = boxw
    pointb[:, :, 3] = boxh

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / res_scal
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / res_scal
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > conf_thresh)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:5000]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_thresh)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    return dets, landms

def test_one(img_path, dnet, snet, minface, dir=False):
    img_mat = cv2.imread(img_path, cv2.IMREAD_COLOR)
    im_h, im_w, _ = img_mat.shape
    face_rect, key_points = detect_one_img(dnet, img_mat, minface)
    for box, lands in zip(face_rect, key_points):
        new_box = expand_facebox(box, im_w, im_h)#人脸框四周扩充
        # new_box = np.zeros(4, dtype=np.int32)
        # new_box[0] = 364
        # new_box[1] = 1032
        # new_box[2] = 1090
        # new_box[3] = 1519

        face_roi = img_mat[new_box[1]:new_box[3], new_box[0]:new_box[2], :]
        roi_process = img_process(face_roi)  # 数据处理，转为网络输入的形式
        # b, c, h, w = roi_process.shape
        # save_feature_channel("txt/imgp.txt", roi_process, b, c, h, w)

        out = snet(roi_process)
        out = torch.sigmoid(out)
        # global shadow_type
        shadow_type = out[0, 0].item()
        showscore = np.round(shadow_type, 4)

        posx = int(new_box[0])
        posy = int(new_box[1])
        cv2.putText(img_mat, str(showscore), (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 255), 4)
        # if shadow_type <= 0.35:
        #     cv2.putText(img_mat, "normal", (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 255), 4)
        # if shadow_type > 0.35:
        #     cv2.putText(img_mat, "shadow", (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 255), 4)

        cv2.rectangle(img_mat, (new_box[0], new_box[1]), (new_box[2], new_box[3]), (0, 255, 0), 4)

    if dir:
        return img_mat, shadow_type
    else:
        cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        cv2.imshow('result', img_mat)
        cv2.waitKey(0)

def test_dir(imdir, savedir, net1, net2, min_face=60):
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    for root, dirs, files in os.walk(imdir):
        for file in files:
            root = root.replace('\\', '/')
            imgpath = root + "/" + file
            savepath = savedir + "/" + file
            saveimg, _type = test_one(imgpath, net1, net2, min_face, dir=True)
            cv2.imshow('result', saveimg)
            cv2.waitKey(1)
            if _type < 0.1:
                img = cv2.imread(imgpath)
                cv2.imwrite(savepath, img)

def get_face_dirs(imgdirs, savedirs, dnet):
    for root, dirs, files in os.walk(imgdirs):
        for file in files:
            imgpath = root + "/" + file
            savepath = savedirs + "/" + file
            img_mat = cv2.imread(imgpath, cv2.IMREAD_COLOR)
            im_h, im_w, _ = img_mat.shape
            face_rect, key_points = detect_one_img(dnet, img_mat, 60)
            for box, lands in zip(face_rect, key_points):
                new_box = expand_facebox(box, im_w, im_h)  # 人脸框四周扩充
                face_roi = img_mat[new_box[1]:new_box[3], new_box[0]:new_box[2], :]
                cv2.imwrite(savepath, face_roi)

def change_score(score=0.5):
    cha = np.zeros(5, dtype=np.float32)
    cha[0] = abs(score - 0.0)
    cha[1] = abs(score - 0.25)
    cha[2] = abs(score - 0.5)
    cha[3] = abs(score - 0.75)
    cha[4] = abs(score - 1.0)
    index = np.argmin(cha)
    ret_score = 0.5
    if index == 0:
        ret_score = 0.0
    if index == 1:
        ret_score = 0.25
    if index == 2:
        ret_score = 0.5
    if index == 3:
        ret_score = 0.75
    if index == 4:
        ret_score = 1.0

    return ret_score

def create_train_samples(imdir, savedir, net1, net2):
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    for root, dirs, files in os.walk(imdir):
        for file in files:
            # filetxt.write(file + ":     ")
            root = root.replace('\\', '/')
            imgpath = root + "/" + file
            img_mat = cv2.imread(imgpath, cv2.IMREAD_COLOR)
            saveimg, _score = test_one(imgpath, net1, net2, minface=60, dir=True)
            lab = change_score(_score)
            if lab == 0.0:
                savepath = savedir + "/0/" + file
            if lab == 0.25:
                savepath = savedir + "/1/" + file
            if lab == 0.5:
                savepath = savedir + "/2/" + file
            if lab == 0.75:
                savepath = savedir + "/3/" + file
            if lab == 1.0:
                savepath = savedir + "/4/" + file
            cv2.imwrite(savepath, img_mat)
            cv2.imshow('result', saveimg)
            cv2.waitKey(1)

def test_one_nodet(img_path, net, dir=False):
    img_mat = cv2.imread(img_path, cv2.IMREAD_COLOR)
    im_h, im_w, _ = img_mat.shape

    roi_process = img_process(img_mat)  # 数据处理，转为网络输入的形式
    out = net(roi_process)
    out = torch.sigmoid(out)
    # global shadow_type
    shadow_type = out[0, 0].item()
    showscore = np.round(shadow_type, 4)

    posx = int(0)
    posy = int(0.2 * im_h)
    cv2.putText(img_mat, str(showscore), (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 255), 4)

    if dir:
        return img_mat, shadow_type
    else:
        cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        cv2.imshow('result', img_mat)
        cv2.waitKey(0)

def create_train_samples_nodet(imdir, savedir, net):
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    for root, dirs, files in os.walk(imdir):
        for file in files:
            # filetxt.write(file + ":     ")
            root = root.replace('\\', '/')
            lab, imname = file.split("_")
            imgpath = root + "/" + file
            img_mat = cv2.imread(imgpath, cv2.IMREAD_COLOR)
            saveimg, _score = test_one_nodet(imgpath, net, dir=True)
            if _score >= 0.0 and _score < 0.07:
                sfile = "0_" + imname
                savepath = savedir + "/0/" + sfile
            if _score >= 0.07 and _score < 0.22:
                sfile = "1_" + imname
                savepath = savedir + "/1/" + sfile
            if _score >= 0.22 and _score < 0.6:
                sfile = "2_" + imname
                savepath = savedir + "/2/" + sfile
            if _score >= 0.6 and _score <= 1.0:
                sfile = "3_" + imname
                savepath = savedir + "/3/" + sfile
            # if _score >= 0.0 and _score < 0.1:
            #     sfile = "4" + imname
            #     savepath = savedir + "/4/" + sfile
            cv2.imwrite(savepath, img_mat)
            cv2.imshow('result', saveimg)
            cv2.waitKey(1)

if __name__ == "__main__":
    snet = NeckShadowNet()  # 需要修改
    s_path = "weights/NeckShadow_0225.pth"  # 需要修改
    s_dict = torch.load(s_path, map_location=lambda storage, loc: storage)
    snet.load_state_dict(s_dict)
    snet.eval()
    snet = snet.to(device)
    saveparams = pytorch_to_dpcoreParams(snet)
    saveparams.forward("NeckShadow_param_cfg.h", "NeckShadow_param_src.h")

    dnet = mobilev3Fpn_small(cfg=cfg)  # 需要修改
    d_path = "detector/mobilev3Fpn_0810_250.pth"  # 需要修改
    d_dict = torch.load(d_path, map_location=lambda storage, loc: storage)
    dnet.load_state_dict(d_dict)
    dnet.eval()
    dnet = dnet.to(device)

    imgpath = "imgs/10f53e703d6211eba12000163e009f55_1.png"
    savepath = "result/res.jpg"
    min_face = 60
    test_one(imgpath, dnet, snet, min_face, dir=False)
    # test_one_nodet(imgpath, snet, dir=False)

    imgdir = "D:/data/imgs/facePicture/shadow/train3"
    savedir = "D:/data/imgs/facePicture/shadow/train4"
    # test_dir(imgdir, savedir, dnet, snet, min_face)
    # get_face_dirs(imgdir, savedir, dnet)
    # create_train_samples(imgdir, savedir, dnet, snet)
    # create_train_samples_nodet(imgdir, savedir, snet)
    print("done")











