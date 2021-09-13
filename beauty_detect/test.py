import torch
import math
import numpy as np
import os
import cv2
import imutils
device = torch.device("cpu")

# retinaface
from retinaface.config import cfg_mnet as cfg
from retinaface.create_anchors import PriorBox
from retinaface.mobilev3_face import mobilev3Fpn_small
from retinaface.retinaface_utils import decode, decode_landm
from retinaface.nms import py_cpu_nms

#beauty
from slim_net import BeautyDetectNet
from test_config import configt as testcfg
from load_data import pytorch_to_dpcoreParams
import time
import random

from slim_net import save_feature_channel

def face_inference(faceNet, img_data, minface, imgresize):
    rgb_mean = cfg['rgb_mean']  # bgr order
    std_mean = cfg['std_mean']
    conf_thresh = 0.5
    nms_thresh = 0.3
    img = np.float32(img_data)
    im_shape = img.shape
    im_size_max = np.max(im_shape[0:2])
    if minface > 0:
        res_scal = 20.0 / float(minface)
    else:
        res_scal = 1.0 * imgresize / im_size_max
    img = cv2.resize(img, None, None, fx=res_scal, fy=res_scal, interpolation=cv2.INTER_LINEAR)

    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    scale = scale.to(device)

    # 减去均值转成numpy
    im_height, im_width, _ = img.shape
    img -= rgb_mean
    img /= std_mean
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)

    # from mobilev3_face import save_feature_channel
    # b, c, h, w = img.shape
    # save_feature_channel("feature/python/img.txt", img, b, c, h, w)

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
    dets = dets[keep, :]
    landms = landms[keep]

    return dets, landms


def img_process(img):
    """将输入图片转换成网络需要的tensor
            Args:
                img_path: 人脸图片路径
            Returns:
                tensor： img(batch, channel, width, height)
    """
    im = cv2.resize(img, (testcfg.img_width, testcfg.img_height), interpolation=cv2.INTER_CUBIC)
    im = im.astype(np.float32)
    im = im / 255.0
    im = im.transpose(2, 0, 1)
    im = torch.from_numpy(im)
    im = im.unsqueeze(0)
    im = im.to(device)
    return im

def crop_facebox(rect, imgw, imgh):
    bx = rect[0]
    by = rect[1]
    bw = rect[2] - rect[0]
    bh = rect[3] - rect[1]

    # #  face big
    # nbx1 = bx - 0.4 * bw
    # nby1 = by - 0.1 * bh
    # nbx2 = nbx1 + 1.8 * bw
    # nby2 = nby1 + 1.45 * bh

    #  face small
    nbx1 = bx - 0.1 * bw
    nby1 = by - 0.1 * bh
    nbx2 = nbx1 + 1.2 * bw
    nby2 = nby1 + 1.2 * bh

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

def show_CropFace(imgp, fnet):
    img_mat = cv2.imread(imgp, cv2.IMREAD_COLOR)
    imgh, imgw, _ = img_mat.shape
    face_rect, key_points = face_inference(fnet, img_mat, 0, 480)
    for box, lands in zip(face_rect, key_points):
        crop_box = crop_facebox(box, imgw, imgh)
        bx1 = crop_box[0]
        by1 = crop_box[1]
        bx2 = crop_box[2]
        by2 = crop_box[3]
        cv2.rectangle(img_mat, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.imshow('result', img_mat)
    cv2.waitKey(0)

def test_img(imgp, fnet, bnet, minface, imgresize, dir=False):
    img_mat = cv2.imread(imgp, cv2.IMREAD_COLOR)
    imgh, imgw, _ = img_mat.shape
    maxwh = max(imgh, imgw)
    daxiao = int(maxwh / 1000.0 * 1.5)
    cuxi = int(maxwh / 1000.0 * 2.0)
    face_rect, key_points = face_inference(fnet, img_mat, minface, imgresize)
    if face_rect.shape[0] > 0:
        ii = 0
        for frect, fpoint in zip(face_rect, key_points):
            ii += 1
            beauty_box = crop_facebox(frect, imgw, imgh)
            beauty_roi = img_mat[beauty_box[1]:beauty_box[3], beauty_box[0]:beauty_box[2], :]
            beauty_img = img_process(beauty_roi)

            output = bnet(beauty_img)
            shadow_type = output[0, 0].item()
            showscore = np.round(shadow_type, 4)
            cv2.rectangle(img_mat, (beauty_box[0], beauty_box[1]), (beauty_box[2], beauty_box[3]), (255, 0, 0), cuxi)

            posx = int(beauty_box[0])
            posy = int(beauty_box[1] + 0.25 * (beauty_box[3] - beauty_box[1]))
            cv2.putText(img_mat, str(showscore), (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 255), daxiao)

    if dir:
        return img_mat, showscore
    else:
        cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        cv2.imshow('result', img_mat)
        cv2.waitKey(0)

def test_dir(imgdir, savedir, fnet, bnet, minface, imgresize):
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            root = root.replace('\\', '/')
            imgpath = root + "/" + file
            savepath = savedir + "/" + file
            saveimg, score = test_img(imgpath, fnet, bnet, minface, imgresize, dir=True)
            cv2.imshow('result', saveimg)
            cv2.waitKey(1)
            cv2.imwrite(savepath, saveimg)
            # if score < 0.1:
            #     img = cv2.imread(imgpath)
            #     cv2.imwrite(savepath, img)

def test_img2(imgp, gnet):
    img_mat = cv2.imread(imgp, cv2.IMREAD_COLOR)
    imgh, imgw, _ = img_mat.shape
    glass_img = img_process(img_mat)
    b, c, h, w = glass_img.shape
    save_feature_channel("txt/p/img.txt", glass_img, b, c, h, w)

    outputs = gnet(glass_img)
    _, prediction = torch.max(outputs.data, 1)

    posx = 0
    posy = int(0.25 * imgw)
    if prediction == 0:
        cv2.putText(img_mat, "No glasses", (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 1)
    if prediction == 1:
        cv2.putText(img_mat, "slim galsses", (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 1)
    if prediction == 2:
        cv2.putText(img_mat, "wide glasses", (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 255), 1)
    if prediction == 3:
        cv2.putText(img_mat, "sun galsses", (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 0), 1)

    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.imshow('result', img_mat)
    cv2.waitKey(1)
    return prediction

def test_dir2(imgdir, savedir, gnet):
    for root, dirs, files in os.walk(imgdir):
        img_num = 0
        for f in files:
            img_num += 1
            if img_num % 1 == 0:
                img_path = os.path.join(root, f)
                path_new = img_path.replace('\\', '/')

                img_save = os.path.join(savedir, f)
                save_new = img_save.replace('\\', '/')
                outc = test_img2(path_new, gnet)
                if outc == 0:
                    img_mat = cv2.imread(path_new, cv2.IMREAD_COLOR)
                    cv2.imwrite(save_new, img_mat)

def crop_face_dir(dnet, imgdir, savedir):
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            root = root.replace("\\", "/")
            rootsplit = root.split("/")
            zidir = rootsplit[-1]
            imgpath = root + "/" + file
            img_mat = cv2.imread(imgpath, cv2.IMREAD_COLOR)
            imgh, imgw, _ = img_mat.shape
            face_rect, key_points = face_inference(dnet, img_mat, 0, 480)
            num = 0
            for box, lands in zip(face_rect, key_points):
                crop_box = crop_facebox(box, imgw, imgh)
                bx1 = crop_box[0]
                by1 = crop_box[1]
                bx2 = crop_box[2]
                by2 = crop_box[3]
                face_roi = img_mat[by1:by2, bx1:bx2, :]
                # resize_roi = cv2.resize(face_roi, (256, 256), interpolation=cv2.INTER_LINEAR)
                if face_rect.shape[0] == 1:
                    savepath = savedir + "/" + file
                else:
                    savepath = savedir + "/" + str(num) + file
                cv2.imwrite(savepath, face_roi)
                cv2.rectangle(img_mat, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            cv2.imshow('result', img_mat)
            cv2.waitKey(1)

if __name__ == "__main__":
    facenet = mobilev3Fpn_small(cfg=cfg)   #需要修改
    face_weight = "weights/detect.pth"   #需要修改
    face_dict = torch.load(face_weight, map_location=lambda storage, loc: storage)
    facenet.load_state_dict(face_dict)
    facenet.eval()
    print('Finished loading face model!')
    facenet = facenet.to(device)

    beautynet = BeautyDetectNet()  # 需要修改
    beauty_weight = "weights/Beauty_250.pth"  # 需要修改
    beauty_dict = torch.load(beauty_weight, map_location=lambda storage, loc: storage)
    beautynet.load_state_dict(beauty_dict)
    beautynet.eval()
    print('Finished loading beauty model!')
    beautynet = beautynet.to(device)
    # saveparams = pytorch_to_dpcoreParams(beautynet)
    # saveparams.forward("Beauty_param_cfg.h", "Beauty_param_src.h")

    img_path = "D:/data/imgs/makeup/pan/beauty/3ff01c18f29811eb957f00163e0070b6.jpg"
    save_path = "save/29.jpg"
    # show_CropFace(img_path, facenet)
    test_img(img_path, facenet, beautynet, 0, 640)
    # test_img2(img_path, glassnet)

    img_dir = "D:/data/imgs/makeup/test/imgs"
    save_dir = "D:/data/imgs/makeup/test/results"
    # test_dir(img_dir, save_dir, facenet, beautynet, 0, 320)
    # test_dir2(img_dir, save_dir, glassnet)

    # crop_face_dir(facenet, img_dir, save_dir)