import torch
import math
import numpy as np
import os
import cv2
import imutils
import random
import shutil

from slim_net import PoseNet
from myconfig import config as testconf

from detector.create_anchors import PriorBox
from detector.config import cfg_slimNet3 as cfg
from detector.face_net import FaceDetectSlimNet
from detector.retinaface_utils import decode, decode_landm
from detector.nms import py_cpu_nms
device = "cpu"

def expand_facebox(rect, imgw, imgh):
    bx = rect[0]
    by = rect[1]
    bw = rect[2] - rect[0]
    bh = rect[3] - rect[1]

    # pose_shouder
    nbx1 = bx - 1.1 * bw  # 0.4
    nby1 = by - 0.28 * bh  # 0.28
    nbx2 = nbx1 + 3.2 * bw  # 1.8
    nby2 = nby1 + 2.7 * bh  # 1.7

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

def test_one_nodet(img_path, snet, dir=False):
    img_mat = cv2.imread(img_path, cv2.IMREAD_COLOR)
    im_h, im_w, _ = img_mat.shape
    imgt = img_process(img_mat)

    out = snet(imgt)
    out = torch.sigmoid(out)
    head = np.around(out[0, 0].item(), 3)
    pose = np.around(out[0, 1].item(), 3)
    shouder = np.around(out[0, 2].item(), 3)

    posx = int(im_w * 0.15)
    posy1 = int(im_h * 0.25)
    posy2 = int(im_h * 0.45)
    posy3 = int(im_h * 0.65)
    cv2.putText(img_mat, str(head), (posx, posy1), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 4)
    cv2.putText(img_mat, str(pose), (posx, posy2), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 4)
    cv2.putText(img_mat, str(shouder), (posx, posy3), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 4)

    if dir:
        return img_mat, head, pose, shouder
    else:
        cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        cv2.imshow('result', img_mat)
        cv2.waitKey(0)

def test_one(img_path, dnet, snet, minface, dir=False):
    img_mat = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # img_mat = img_path
    im_h, im_w, _ = img_mat.shape
    tl = round(0.002 * (im_h + im_w) / 2) + 1
    face_rect, key_points = detect_one_img(dnet, img_mat, minface)
    for box, lands in zip(face_rect, key_points):
        new_box = expand_facebox(box, im_w, im_h)#人脸框四周扩充
        face_roi = img_mat[new_box[1]:new_box[3], new_box[0]:new_box[2], :]
        face_roi = img_process(face_roi)
        out = snet(face_roi)
        out = torch.sigmoid(out)
        head = np.around(out[0, 0].item(), 3)
        pose = np.around(out[0, 1].item(), 3)
        shouder = np.around(out[0, 2].item(), 3)

        posx = int(im_w * 0.15)
        posy1 = int(im_h * 0.25)
        posy2 = int(im_h * 0.45)
        posy3 = int(im_h * 0.65)
        cv2.putText(img_mat, str(head), (posx, posy1), cv2.FONT_HERSHEY_COMPLEX, tl, (0, 0, 255), 4)
        cv2.putText(img_mat, str(pose), (posx, posy2), cv2.FONT_HERSHEY_COMPLEX, tl, (0, 0, 255), 4)
        cv2.putText(img_mat, str(shouder), (posx, posy3), cv2.FONT_HERSHEY_COMPLEX, tl, (0, 0, 255), 4)
        cv2.rectangle(img_mat, (new_box[0], new_box[1]), (new_box[2], new_box[3]), (0, 255, 255), thickness=tl, lineType=8)

    if dir:
        return img_mat, head, pose, shouder
    else:
        cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        cv2.imshow('result', img_mat)
        cv2.waitKey(0)

def test_dir(imgdir, savedir, dnet, qnet, mface):
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    # jieguo = open("D:/data/imgs/facePicture/pose_person/pose_shouder/results.txt", mode="w+")
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            root = root.replace("\\", "/")
            rootsplit = root.split("/")
            zidir = rootsplit[-1]
            imgpath = root + "/" + file
            imgsave, head, pose, shouder = test_one(imgpath, dnet, qnet, mface, dir=True)
            savedata = file + " " + str(head) + " " + str(pose) + " " + str(shouder)
            # jieguo.write(savedata)
            # jieguo.write("\n")
            savepath = savedir + "/" + file
            cv2.imwrite(savepath, imgsave)
            cv2.imshow('result', imgsave)
            cv2.waitKey(1)

def test_dir_nodet(imgdir, savedir, net):
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    # jieguo = open("D:/data/imgs/facePicture/pose_person/pose_shouder/results.txt", mode="w+")
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            root = root.replace("\\", "/")
            rootsplit = root.split("/")
            zidir = rootsplit[-1]
            imgpath = root + "/" + file
            imgsave, head, pose, shouder = test_one_nodet(imgpath, net, dir=True)
            savedata = file + " " + str(head) + " " + str(pose) + " " + str(shouder)
            # jieguo.write(savedata)
            # jieguo.write("\n")
            savepath = savedir + "/" + file
            cv2.imwrite(savepath, imgsave)
            cv2.imshow('result', imgsave)
            cv2.waitKey(1)
            # shutil.move(imgpath, savepath)
    # jieguo.close()

def score2class(score):
    cla = 0
    if score >= 0.0 and score < 0.15:
        cla = 0
    if score >= 0.15 and score < 0.45:
        cla = 1
    if score >= 0.45 and score < 0.8:
        cla = 2
    if score >= 0.8 and score <= 1:
        cla = 3
    return cla

def renameLabel_dir_nodet(imdir, net):
    # cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    for root, dirs, files in os.walk(imdir):
        for file in files:
            filesplit = file.split("_")
            filename = filesplit[-1]
            if len(filesplit) > 4:
                filename = filesplit[-2] + filesplit[-1]
            root = root.replace('\\', '/')
            imgpath = root + "/" + file
            saveimg, _hscore, _pscore, _sscore = test_one_nodet(imgpath, net, dir=True)
            hcla = score2class(_hscore)
            pcla = score2class(_pscore)
            scla = score2class(_sscore)
            hs = str(hcla)
            ps = str(pcla)
            ss = str(scla)

            savepath = root + "/" + hs + "_" + ps + "_" + ss + "_" + filename

            os.rename(imgpath, savepath)

def get_face_dirs(imgdirs, savedirs, dnet):
    for root, dirs, files in os.walk(imgdirs):
        for file in files:
            root = root.replace('\\', '/')
            imgname, houzui = file.split(".")
            imgpath = root + "/" + file
            # savepath = savedirs + "/" + imgname + "_0." + houzui
            savepath = savedirs + "/" + file
            img_mat = cv2.imread(imgpath, cv2.IMREAD_COLOR)
            im_h, im_w, _ = img_mat.shape
            face_rect, key_points = detect_one_img(dnet, img_mat, 60)
            for box, lands in zip(face_rect, key_points):
                new_box = expand_facebox(box, im_w, im_h)  # 人脸框四周扩充
                face_roi = img_mat[new_box[1]:new_box[3], new_box[0]:new_box[2], :]
                cv2.imwrite(savepath, face_roi)

def test_camORvideo(vpath, spath, dnet, qnet, min_face):
    cap = cv2.VideoCapture(0)  # 从摄像头中取得视频
    # 获取视频播放界面长宽
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    # 定义编码器 创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # Be sure to use the lower case
    outvideo = cv2.VideoWriter(spath, fourcc, 20.0, (640, 640))

    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    while (cap.isOpened()):
        # 读取帧摄像头
        ret, frame = cap.read()
        if ret == True:
            res_frame, hs, ps, ss = test_one(frame, dnet, qnet, min_face, dir=True)
            outvideo.write(res_frame)
            cv2.imshow('result', res_frame)
            cv2.waitKey(1)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
        else:
            break
    # 释放资源
    outvideo.release()
    cap.release()
    cv2.destroyAllWindows()

def crop_wanted_area(imgdir, savedir, dnet, minface):
    # cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    for root, dirs, files in os.walk(imgdir):
        ii = 0
        for file in files:
            if file.endswith("txt"):
                continue
            else:
                hz = file.split(".")[-1]
                imgpath = imgdir + "/" + file
                imgdata = cv2.imread(imgpath, cv2.IMREAD_COLOR)
                if imgdata is None:
                    print(imgpath)
                # show_rot = imgdata.copy()
                imh, imw, _ = imgdata.shape
                face_rect, key_points = detect_one_img(dnet, imgdata, minface)
                for box, lands in zip(face_rect, key_points):
                    rect = expand_facebox(box, imw, imh)
                    roi = imgdata[rect[1]:rect[3], rect[0]:rect[2], :]

                    # resize
                    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
                    interp_method = interp_methods[random.randrange(5)]
                    roi = cv2.resize(roi, (256, 256), interpolation=interp_method)

                    savename = "addmb" + str(ii) + ".jpg"
                    savepath = savedir + "/" + savename
                    # savepath = savedir + "/" + file
                    ii += 1

                    cv2.imwrite(savepath, roi)
            # cv2.imshow('result', show_rot)
            # cv2.waitKey(1)

def expand_facebox2(rect, imgw, imgh):
    bx = rect[0]
    by = rect[1]
    bw = rect[2] - rect[0]
    bh = rect[3] - rect[1]

    crop_ratio = [[1.0, 0.28, 3.0, 2.5], [1.0, 0.28, 3.0, 2.3], [0.8, 0.28, 2.6, 2.1], [0.9, 0.28, 2.8, 2.0], [0.7, 0.28, 2.4, 1.7]]
    num = len(crop_ratio)
    cropbox = []
    for i in range(num):
        ratio = crop_ratio[i]
        # pose_shouder
        nbx1 = bx - ratio[0] * bw  # 0.4
        nby1 = by - ratio[1] * bh  # 0.28
        nbx2 = nbx1 + ratio[2] * bw  # 1.8
        nby2 = nby1 + ratio[3] * bh  # 1.7

        pp = np.zeros(4, dtype=np.int32)
        rx1 = max(nbx1, 0)
        ry1 = max(nby1, 0)
        rx2 = min(nbx2, imgw)
        ry2 = min(nby2, imgh)

        pp[0] = rx1
        pp[1] = ry1
        pp[2] = rx2
        pp[3] = ry2
        cropbox.append(pp)
    return cropbox

def crop_wanted_area2(imgdir, savedir, dnet, minface):
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
                for box, lands in zip(face_rect, key_points):
                    rects = expand_facebox2(box, imw, imh)
                    ii = 0
                    for rect in rects:
                        roi = imgdata[rect[1]:rect[3], rect[0]:rect[2], :]

                        # roi = cv2.resize(roi, (256, 256), interpolation=cv2.INTER_LINEAR)
                        savepath = savedir + "/" + iname + str(ii) + ".jpg"
                        cv2.imwrite(savepath, roi)
                        ii += 1

                        # ih, iw, _ = roi.shape
                        # if ih < 256 or iw < 256:
                        #     savepath = savedir + "/" + iname + str(ii) + ".jpg"
                        #     cv2.imwrite(savepath, roi)
                        #     ii += 1

def expand_facebox3(rect, imgw, imgh):
    bx = rect[0]
    by = rect[1]
    bw = rect[2] - rect[0]
    bh = rect[3] - rect[1]

    num = 10
    cropbox = []
    for i in range(num):
        ratiox = random.uniform(1.3, 0.7)
        ratioy = random.uniform(0.25, 0.3)
        ratiow = random.uniform(2.4, 3.6)
        ratioh = random.uniform(1.7, 2.8)
        # pose_shouder
        nbx1 = bx - ratiox * bw  # 0.4
        nby1 = by - ratioy * bh  # 0.28
        nbx2 = nbx1 + ratiow * bw  # 1.8
        nby2 = nby1 + ratioh * bh  # 1.7

        pp = np.zeros(4, dtype=np.int32)
        rx1 = max(nbx1, 0)
        ry1 = max(nby1, 0)
        rx2 = min(nbx2, imgw)
        ry2 = min(nby2, imgh)

        pp[0] = rx1
        pp[1] = ry1
        pp[2] = rx2
        pp[3] = ry2
        cropbox.append(pp)
    return cropbox

def show_crop(imgpath):
    bgr = [150, 165, 175, 185, 200, 210, 220, 235, 255]
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    imgdata = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    imh, imw, _ = imgdata.shape
    face_rect, key_points = detect_one_img(dnet, imgdata, 20)
    for box, lands in zip(face_rect, key_points):
        rects = expand_facebox2(box, imw, imh)
        for rect in rects:
            b = random.choice(bgr)
            g = random.choice(bgr)
            r = random.choice(bgr)
            tl = round(0.002 * (imh + imw) / 2) + 1
            cv2.rectangle(imgdata, (rect[0], rect[1]), (rect[2], rect[3]), (b, g, r), thickness=tl, lineType=8)
    cv2.imshow('result', imgdata)
    cv2.waitKey(0)

if __name__ == "__main__":
    qnet = PoseNet()  # 需要修改
    q_path = "weights/pose_0813.pth"  # 需要修改
    q_dict = torch.load(q_path, map_location=lambda storage, loc: storage)
    qnet.load_state_dict(q_dict)
    qnet.eval()
    qnet = qnet.to(device)

    dnet = FaceDetectSlimNet(cfg=cfg)  # 需要修改
    d_path = "weights/face_slim_0609_250.pth"  # 需要修改
    d_dict = torch.load(d_path, map_location=lambda storage, loc: storage)
    dnet.load_state_dict(d_dict)
    dnet.eval()
    dnet = dnet.to(device)

    imgpath = "test/00ebfc74e9c611eb94a600163e0070b6.jpg"
    savepath = "test/bb"
    min_face = 60
    # test_one(imgpath, dnet, qnet, min_face, dir=False)
    # test_one_nodet(imgpath, qnet, dir=False)
    # show_crop(imgpath)

    imgdir = "D:/data/imgs/facePicture/pose_person/pose_shouder/test4/bad"
    savedir = "D:/data/imgs/facePicture/pose_person/pose_shouder/test4/result/dd"
    txt1 = "D:/data/imgs/facePicture/blur/test/result_1.txt"
    txt2 = "D:/data/imgs/facePicture/blur/test/result_2.txt"
    txtl = "D:/data/imgs/facePicture/blur/test/label.txt"
    test_dir(imgdir, savedir, dnet, qnet, min_face)
    # renameLabel_dir_nodet(imgdir, qnet)

    imgd = "D:/data/imgs/facePicture/pose_person/pose_shouder/train/train0"
    saved = "D:/data/imgs/facePicture/pose_person/pose_shouder/train/copsNormal"
    # test_dir_nodet(imgd, saved, qnet)
    # crop_wanted_area2(imgpath, savepath, dnet, min_face)

    videopath = "test.avi"
    svpath = "res.avi"
    # test_camORvideo(videopath, svpath, dnet, qnet, min_face)











