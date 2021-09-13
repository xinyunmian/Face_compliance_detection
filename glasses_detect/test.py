import torch
import math
import numpy as np
import os
import cv2
import imutils
import shutil
device = torch.device("cpu")

# retinaface
from detector.create_anchors import PriorBox
from detector.config import cfg_slimNet3 as cfg
from detector.face_net import FaceDetectSlimNet
from detector.retinaface_utils import decode, decode_landm
from detector.nms import py_cpu_nms

#eyeglass
from mask_net import eyeGlassNet
from myconfig import config as testcfg
import time
import random
from mask_net import save_feature_channel

def img_process(img):
    """将输入图片转换成网络需要的tensor
            Args:
                img_path: 人脸图片路径
            Returns:
                tensor： img(batch, channel, width, height)
    """
    im = cv2.resize(img, (testcfg.img_width, testcfg.img_height), interpolation=cv2.INTER_CUBIC)
    im = im.astype(np.float32)
    im = (im - testcfg.bgr_mean) / testcfg.bgr_std
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

def get_glasses_rect(left_eyex, left_eyey, right_eyex, right_eyey, imgw, imgh):
    rect_glass = np.zeros(4, dtype=np.int32)
    g_centerx = 0.5 * (left_eyex + right_eyex)
    g_centery = 0.5 * (left_eyey + right_eyey)
    g_w = 2.5 * (right_eyex - left_eyex)
    g_h = 0.6 * g_w

    g_lx = g_centerx - 0.5 * g_w
    if (g_lx < 0):
        g_lx = 0
    g_ly = g_centery - 0.5 * g_h
    if (g_ly < 0):
        g_ly = 0
    g_rx = g_centerx + 0.5 * g_w
    if (g_rx > imgw):
        g_rx = imgw
    g_ry = g_centery + 0.5 * g_h
    if (g_ry > imgh):
        g_ry = imgh

    rect_glass[0] = g_lx
    rect_glass[1] = g_ly
    rect_glass[2] = g_rx
    rect_glass[3] = g_ry

    return rect_glass

def test_img(imgp, fnet, gnet, dir=False):
    plist = imgp.split("/")
    img_name = plist[-1]
    img_mat = cv2.imread(imgp, cv2.IMREAD_COLOR)
    imgh, imgw, _ = img_mat.shape
    face_rect, key_points = detect_one_img(fnet, img_mat, minface=40)
    prediction = 0
    if face_rect.shape[0] > 0:
        ii = 0
        for frect, fpoint in zip(face_rect, key_points):
            ii += 1
            pp = np.zeros(4, dtype=np.int32)
            pp[0] = fpoint[0]
            pp[1] = fpoint[1]
            pp[2] = fpoint[2]
            pp[3] = fpoint[3]
            glass_box = get_glasses_rect(pp[0], pp[1], pp[2], pp[3], imgw, imgh)
            glass_roi = img_mat[glass_box[1]:glass_box[3], glass_box[0]:glass_box[2], :]

            # h, w, c = glass_roi.shape
            # glass_roi = glass_roi.transpose(2, 0, 1)
            # glass_roi = torch.from_numpy(glass_roi)
            # glass_roi = glass_roi.unsqueeze(0)
            # b, c, h, w = glass_img.shape
            # save_feature_channel("txt/p/normimg.txt", glass_img, b, c, h, w)

            ih, iw, _ = glass_roi.shape
            if ih < 64 or iw < 64:
                continue

            glass_img = img_process(glass_roi)
            # b, c, h, w = glass_img.shape
            # save_feature_channel("txt/p/normimg.txt", glass_img, b, c, h, w)

            outputs = gnet(glass_img)
            _, prediction = torch.max(outputs.data, 1)
            cv2.rectangle(img_mat, (glass_box[0], glass_box[1]), (glass_box[2], glass_box[3]), (255, 0, 0), 2)

            posx = int(glass_box[0])
            posy = int(glass_box[1] + 0.25 * (glass_box[3] - glass_box[1]))
            if prediction == 0:
                cv2.putText(img_mat, "No glasses", (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
            if prediction == 1:
                cv2.putText(img_mat, "slim galsses", (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
            if prediction == 2:
                cv2.putText(img_mat, "wide glasses", (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 255), 2)
            if prediction == 3:
                cv2.putText(img_mat, "sun galsses", (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 0), 2)
    if dir:
        return img_mat, prediction
    else:
        cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        cv2.imshow('result', img_mat)
        cv2.waitKey(0)

def test_dir(imgdir, savedir, fnet, gnet):
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    for root, dirs, files in os.walk(imgdir):
        img_num = 0
        for f in files:
            img_num += 1
            if img_num % 1 == 0:
                img_path = os.path.join(root, f)
                path_new = img_path.replace('\\', '/')
                imgsave, pre = test_img(path_new, fnet, gnet, dir=True)
                # if pre == 0:
                #     savepath = savedir + "/0/" + f
                #     shutil.copy(path_new, savepath)
                # if pre == 1:
                #     savepath = savedir + "/1/" + f
                #     shutil.copy(path_new, savepath)
                # if pre == 2:
                #     savepath = savedir + "/2/" + f
                #     shutil.copy(path_new, savepath)
                # if pre == 3:
                #     savepath = savedir + "/3/" + f
                #     shutil.copy(path_new, savepath)

                savepath = savedir + "/" + f
                cv2.imwrite(savepath, imgsave)
                cv2.imshow('result', imgsave)
                cv2.waitKey(1)


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

def get_glasses_rect2(left_eyex, left_eyey, right_eyex, right_eyey, imgw, imgh):
    wh_ratio = [[2.5, 0.6], [2.5, 0.7], [2.7, 0.65], [2.6, 0.55], [2.45, 0.75], [2.7, 0.7], [2.55, 0.55], [2.65, 0.6]]
    # wh_ratio = [[2.5, 0.6], [2.5, 0.7], [2.6, 0.55], [2.55, 0.55], [2.65, 0.6]]
    # wh_ratio = [[2.5, 0.6], [2.55, 0.55], [2.65, 0.6]]

    num = len(wh_ratio)
    cropbox = []
    for i in range(num):
        g_centerx = 0.5 * (left_eyex + right_eyex)
        randch = random.choice([0.48, 0.50, 0.52, 0.54, 0.56, 0.58])
        g_centery = randch * (left_eyey + right_eyey)

        ratio = wh_ratio[i]
        g_w = ratio[0] * (right_eyex - left_eyex)
        g_h = ratio[1] * g_w

        rect_glass = np.zeros(4, dtype=np.int32)

        g_lx = g_centerx - 0.5 * g_w
        if (g_lx < 0):
            g_lx = 0
        g_ly = g_centery - 0.5 * g_h
        if (g_ly < 0):
            g_ly = 0
        g_rx = g_centerx + 0.5 * g_w
        if (g_rx > imgw):
            g_rx = imgw
        g_ry = g_centery + 0.5 * g_h
        if (g_ry > imgh):
            g_ry = imgh

        rect_glass[0] = g_lx
        rect_glass[1] = g_ly
        rect_glass[2] = g_rx
        rect_glass[3] = g_ry
        cropbox.append(rect_glass)

    return cropbox

def crop_glasses_area(imgdir, savedir, dnet, minface):
    # cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            if file.endswith("txt"):
                continue
            else:
                iname, hz = file.split(".")
                imgpath = imgdir + "/" + file
                imgdata = cv2.imread(imgpath, cv2.IMREAD_COLOR)
                if imgdata is None:
                    print(imgpath)
                # show_rot = imgdata.copy()
                imh, imw, _ = imgdata.shape
                face_rect, key_points = detect_one_img(dnet, imgdata, minface)
                lands = key_points[0]
                pp = np.zeros(4, dtype=np.int32)
                pp[0] = lands[0]
                pp[1] = lands[1]
                pp[2] = lands[2]
                pp[3] = lands[3]
                rects = get_glasses_rect2(pp[0], pp[1], pp[2], pp[3], imw, imh)
                ii = 0
                for rect in rects:
                    roi = imgdata[rect[1]:rect[3], rect[0]:rect[2], :]
                    # resize
                    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
                    interp_method = interp_methods[random.randrange(5)]
                    roi = cv2.resize(roi, (128, 64), interpolation=interp_method)
                    savepath = savedir + "/" + iname + str(ii) + ".jpg"
                    cv2.imwrite(savepath, roi)
                    ii += 1

def show_crop(imgpath):
    bgr = [150, 165, 175, 185, 200, 210, 220, 235, 255]
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    imgdata = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    imh, imw, _ = imgdata.shape
    face_rect, key_points = detect_one_img(dnet, imgdata, 20)
    for box, lands in zip(face_rect, key_points):
        pp = np.zeros(4, dtype=np.int32)
        pp[0] = lands[0]
        pp[1] = lands[1]
        pp[2] = lands[2]
        pp[3] = lands[3]
        rects = get_glasses_rect2(pp[0], pp[1], pp[2], pp[3], imw, imh)
        for rect in rects:
            b = random.choice(bgr)
            g = random.choice(bgr)
            r = random.choice(bgr)
            tl = round(0.002 * (imh + imw) / 2) + 1
            cv2.rectangle(imgdata, (rect[0], rect[1]), (rect[2], rect[3]), (b, g, r), thickness=tl, lineType=8)
    cv2.imshow('result', imgdata)
    cv2.waitKey(0)

if __name__ == "__main__":
    device = "cpu"
    glassnet = eyeGlassNet(n_class=4)
    glass_weight = "weights/glasses_300_0827.pth"  # 需要修改
    glass_dict = torch.load(glass_weight, map_location=lambda storage, loc: storage)
    glassnet.load_state_dict(glass_dict)
    glassnet.eval()
    print('Finished loading eyeglass model!')
    glassnet = glassnet.to(device)

    dnet = FaceDetectSlimNet(cfg=cfg)  # 需要修改
    d_path = "weights/face_slim_0609_250.pth"  # 需要修改
    d_dict = torch.load(d_path, map_location=lambda storage, loc: storage)
    dnet.load_state_dict(d_dict)
    dnet.eval()
    dnet = dnet.to(device)

    img_path = "D:/data/imgs/facePicture/glasses/error/3/err(11).jpg"
    save_path = "save/29.jpg"
    # show_crop(img_path)
    test_img(img_path, dnet, glassnet, dir=False)
    # test_img2(img_path, glassnet)

    img_dir = "D:/data/imgs/facePicture/glasses/error/3"
    save_dir = "D:/data/imgs/facePicture/glasses/test_results"
    # test_dir(img_dir, save_dir, dnet, glassnet)
    # test_dir2(img_dir, save_dir, glassnet)
    # crop_glasses_area(img_dir, save_dir, dnet, 60)





