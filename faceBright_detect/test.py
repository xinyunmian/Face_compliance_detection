import torch
import math
import numpy as np
import os
import cv2
import imutils
import random

from model import FaceBrightNet, save_feature_channel
from slim_net import FaceSkinNet, FaceSkinMobileNet
from myconfig import config as testconf

from detector.create_anchors import PriorBox
from detector.config import cfg_slimNet3 as cfg
from detector.face_net import FaceDetectSlimNet
from detector.retinaface_utils import decode, decode_landm
from detector.nms import py_cpu_nms

device = "cuda"
# Define the correct points.128*128
POINTS = np.array([
    [42, 52],
    [92, 52],
    [67, 76],
    [46, 94],
    [86, 94]
], np.int32)

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def expand_facebox(rect, imgw, imgh):
    bx = rect[0]
    by = rect[1]
    bw = rect[2] - rect[0]
    bh = rect[3] - rect[1]

    #  face
    nbx1 = bx - 0.1 * bw
    nby1 = by - 0.1 * bh
    nbx2 = nbx1 + 1.2 * bw
    nby2 = nby1 + 1.1 * bh

    # # neck
    # nbx1 = bx + 0 * bw
    # nby1 = by + 0.72 * bh
    # nbx2 = nbx1 + 1.0 * bw
    # nby2 = nby1 + 0.55 * bh

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
    conf_thresh = 0.5
    nms_thresh = 0.3
    im_shape = img_data.shape
    im_size_max = np.max(im_shape[0:2])
    res_scal = 640 / im_size_max
    # res_scal = 20 / float(minface)
    neww = (int(im_shape[1] * res_scal / 64) + 1) * 64
    newh = (int(im_shape[0] * res_scal / 64) + 1) * 64
    scalw = neww / im_shape[1]
    scalh = newh / im_shape[0]

    img = np.float32(img_data)
    # img = cv2.resize(img, None, None, fx=res_scal, fy=res_scal, interpolation=cv2.INTER_CUBIC)
    img = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_LINEAR)
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    scale = scale.to(device)

    # 减去均值转成numpy
    im_height, im_width, _ = img.shape
    img /= 255.0
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)

    # b, c, h, w = img.shape
    # save_feature_channel("txt/imgp.txt", img, b, c, h, w)

    loc, conf, landms = faceNet(img)  # forward pass

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])

    # boxes = boxes * scale / res_scal

    boxes = boxes * scale
    boxes[:, (0, 2)] = boxes[:, (0, 2)] / scalw
    boxes[:, (1, 3)] = boxes[:, (1, 3)] / scalh

    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)

    # landms = landms * scale1 / res_scal

    landms = landms * scale1
    landms[:, (0, 2, 4, 6, 8)] = landms[:, (0, 2, 4, 6, 8)] / scalw
    landms[:, (1, 3, 5, 7, 9)] = landms[:, (1, 3, 5, 7, 9)] / scalh

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

def test_one(img_mat, dnet, fbnet, minface):
    # img_mat = cv2.imread(imgp, cv2.IMREAD_COLOR)
    im_h, im_w, _ = img_mat.shape
    face_rect, key_points = detect_one_img(dnet, img_mat, minface)
    for box, lands in zip(face_rect, key_points):
        bx1 = box[0]
        by1 = box[1]
        bx2 = box[2]
        by2 = box[3]

        new_box = expand_facebox(box, im_w, im_h)#人脸框四周扩充
        face_roi = img_mat[new_box[1]:new_box[3], new_box[0]:new_box[2], :]
        roi_process = img_process(face_roi)  # 数据处理，转为网络输入的形式
        pdark, pbright, pyinyang = fbnet(roi_process)
        _, p_d = torch.max(pdark.data, 1)
        _, p_b = torch.max(pbright.data, 1)
        _, p_y = torch.max(pyinyang.data, 1)

        posx = int(box[0])
        posy = int(box[1])
        txtshow = str(p_d.item()) + " " + str(p_b.item()) + " " + str(p_y.item()) + " "
        cv2.putText(img_mat, txtshow, (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 255), 6)

        cv2.rectangle(img_mat, (new_box[0], new_box[1]), (new_box[2], new_box[3]), (0, 255, 0), 2)
    return img_mat, p_d, p_b, p_y

def show_FaceTypeScore(img_mat, bright, dark, yy, skin, ratio, cuxi):
    imh, imw, _ = img_mat.shape
    posx = int(0.01 * imw)
    posyb = int(0.15 * imh)
    posyd = int(0.3 * imh)
    posyy = int(0.45 * imh)
    posys = int(0.6 * imh)
    posyr = int(0.75 * imh)
    txtshowb = "bright: " + str(bright)
    txtshowd = "dark: " + str(dark)
    txtshowy = "illumination: " + str(yy)
    txtshows = "skin color: " + str(skin)
    txtshowr = "contrast: " + str(ratio)
    cv2.putText(img_mat, txtshowb, (posx, posyb), cv2.FONT_HERSHEY_COMPLEX, cuxi, (255, 0, 0), 4)
    cv2.putText(img_mat, txtshowd, (posx, posyd), cv2.FONT_HERSHEY_COMPLEX, cuxi, (255, 0, 0), 4)
    cv2.putText(img_mat, txtshowy, (posx, posyy), cv2.FONT_HERSHEY_COMPLEX, cuxi, (255, 0, 0), 4)
    cv2.putText(img_mat, txtshows, (posx, posys), cv2.FONT_HERSHEY_COMPLEX, cuxi, (255, 0, 0), 4)
    cv2.putText(img_mat, txtshowr, (posx, posyr), cv2.FONT_HERSHEY_COMPLEX, cuxi, (255, 0, 0), 4)
    return img_mat

def test_FaceSkin_one(img_mat, dnet, fbnet):
    # img_mat = cv2.imread(imgp, cv2.IMREAD_COLOR)
    im_h, im_w, _ = img_mat.shape
    tl = round(0.002 * (im_h + im_w) / 2) + 1
    face_rect, key_points = detect_one_img(dnet, img_mat, 60)
    for box, lands in zip(face_rect, key_points):
        new_box = expand_facebox(box, im_w, im_h)#人脸框四周扩充
        # new_box = np.zeros(4, dtype=np.int32)
        # new_box[0] = 56
        # new_box[1] = 50
        # new_box[2] = 290
        # new_box[3] = 368

        face_roi = img_mat[new_box[1]:new_box[3], new_box[0]:new_box[2], :]
        roi_process = img_process(face_roi)  # 数据处理，转为网络输入的形式

        # b, c, h, w = roi_process.shape
        # save_feature_channel("txt/p/img.txt", roi_process, b, c, h, w)

        out = fbnet(roi_process)
        out = torch.sigmoid(out)
        bright = out[0][0].item()
        dark = out[0][1].item()
        yy = out[0][2].item()
        skin = out[0][3].item()
        ratio = out[0][4].item()
        bright = np.around(bright, 2)
        dark = np.around(dark, 2)
        yy = np.around(yy, 2)
        skin = np.around(skin, 2)
        ratio = np.around(ratio, 2)

        # show_FaceTypeScore(img_mat, bright, dark, yy, skin, ratio, tl)

        # cv2.rectangle(img_mat, (new_box[0], new_box[1]), (new_box[2], new_box[3]), (0, 255, 0), tl)
    return img_mat, bright, dark, yy, skin, ratio

def test_dir(imdir, savedir, net1, net2, min_face):
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    for root, dirs, files in os.walk(imdir):
        for file in files:
            if file.endswith("jpg"):
                root = root.replace('\\', '/')
                imgpath = root + "/" + file
                imgdir, imgname = os.path.split(imgpath)
                savepath = os.path.join(savedir, imgname)
                img_mat = cv2.imread(imgpath, cv2.IMREAD_COLOR)
                saveimg, dark, bright, yinyang = test_one(img_mat, net1, net2, min_face)
                cv2.imshow('result', saveimg)
                cv2.waitKey(1)
                cv2.imwrite(savepath, saveimg)

def test_rename_dir(imdir, net1, net2):
    skins = 0
    for root, dirs, files in os.walk(imdir):
        for file in files:
            imgpath = imdir + "/" + file
            imgname, imghz = file.split(".")
            img_mat = cv2.imread(imgpath)
            if img_mat is not None:
                saveimg, bri, dar, yiya, ski, rat = test_FaceSkin_one(img_mat, net1, net2)
                if bri > 0.5 or dar > 0.5 or yiya > 0.5 or ski > 0.5 or rat > 0.5:
                    skins = 1
                savename = imdir + "/" + imgname + "_" + str(skins) + "." + imghz
                os.rename(imgpath, savename)

def get_face_dirs(imgdirs, savedirs, dnet):
    for root, dirs, files in os.walk(imgdirs):
        for file in files:
            rootsplit = root.replace('\\', '/').split("/")
            dir = rootsplit[-1]
            # imgpath = imgdirs + "/" + dir + "/" +file
            savepath = savedirs + "/" + file
            imgpath = root + "/" + file
            img_mat = cv2.imread(imgpath, cv2.IMREAD_COLOR)
            im_h, im_w, _ = img_mat.shape
            face_rect, key_points = detect_one_img(dnet, img_mat, 60)
            for box, lands in zip(face_rect, key_points):
                new_box = expand_facebox(box, im_w, im_h)  # 人脸框四周扩充
                # cv2.rectangle(img_mat, (new_box[0], new_box[1]), (new_box[2], new_box[3]), (0, 255, 0), 2)
                # cv2.imshow('result', img_mat)
                # cv2.waitKey(0)
                face_roi = img_mat[new_box[1]:new_box[3], new_box[0]:new_box[2], :]
                cv2.imwrite(savepath, face_roi)
                # cv2.waitKey(1000)

def test_camORvideo(vpath, spath, net1, net2, min_face):
    cap = cv2.VideoCapture(0)  # 从摄像头中取得视频
    # 获取视频播放界面长宽
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    # 定义编码器 创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # Be sure to use the lower case
    outvideo = cv2.VideoWriter(spath, fourcc, 20.0, (width, height))

    cv2.namedWindow('result1', cv2.WINDOW_NORMAL)
    while (cap.isOpened()):
        # 读取帧摄像头
        ret, frame = cap.read()
        if ret == True:
            res_frame = test_FaceSkin_one(frame, net1, net2)
            # res_frame, dark, bright, yinyang = test_one(frame, net1, net2, min_face)
            outvideo.write(res_frame)
            cv2.imshow('result1', res_frame)
            cv2.waitKey(1)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
        else:
            break
    # 释放资源
    outvideo.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    dnet = FaceDetectSlimNet(cfg=cfg)  # 需要修改
    d_path = "weights/face_slim_0609_250.pth"  # 需要修改
    d_dict = torch.load(d_path, map_location=lambda storage, loc: storage)
    dnet.load_state_dict(d_dict)
    dnet.eval()
    dnet = dnet.to(device)

    # 人脸光照
    FBnet = FaceSkinMobileNet()  # 需要修改
    pretrained_ag = "weights/FaceSkin_Mobile_300.pth"  # 需要修改
    ag_dict = torch.load(pretrained_ag, map_location=lambda storage, loc: storage)
    if "state_dict" in ag_dict.keys():
        ag_dict = remove_prefix(ag_dict['state_dict'], 'module.')
    else:
        ag_dict = remove_prefix(ag_dict, 'module.')
    check_keys(FBnet, ag_dict)
    FBnet.load_state_dict(ag_dict, strict=False)
    FBnet.eval()
    print('Finished loading model!')
    FBnet = FBnet.to(device)


    imgpath = "imgs/cc.png"
    savepath = "D:/codes/project/ageTest/result/af93c266db7b11eaad2600163e0070b6.jpg"
    # min_face = 20
    # im = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    # # simg, dark, bright, yinyang = test_one(im, Dnet, FBnet, min_face)
    # simg = test_FaceSkin_one(im, dnet, FBnet)
    # cv2.namedWindow('result1', cv2.WINDOW_NORMAL)
    # cv2.imshow('result1', simg)
    # cv2.waitKey(0)

    videop = ""
    savep = "result/1.mp4"
    # test_camORvideo(videop, savep, Dnet, FBnet, min_face)

    dir = "D:/wx/1117"
    save = "D:/data/imgs/facePicture/shadow/train/nm"
    test_rename_dir(dir, dnet, FBnet)
    # test_dir(dir, save, Dnet, AGnet, min_face)
    # get_face_dirs(dir, save, Dnet)









