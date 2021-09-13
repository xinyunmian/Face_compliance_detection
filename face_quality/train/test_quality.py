import torch
import math
import numpy as np
import os
import cv2
import imutils
import random
import shutil

from slim_net import FaceQualityNet, FaceQualitySlim
from myconfig import config as testconf
from load_data import pytorch_to_dpcoreParams, save_feature_channel, get_patches, get_patches_augment

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

    # face
    nbx1 = bx - 0 * bw #0.1,0.1,1.2,1.1
    nby1 = by - 0 * bh
    nbx2 = nbx1 + 1 * bw
    nby2 = nby1 + 1 * bh

    # neck
    # # randid = random.choice([1, 2, 3, 4, 5, 6])
    # # sx1, sy1, sx2, sy2 = rand_ratio(randid)
    # sx1 = -0.03
    # sy1 = 0.72
    # sx2 = 1.06
    # sy2 = 0.65
    #
    # nbx1 = bx + sx1 * bw
    # nby1 = by + sy1 * bh
    # nbx2 = nbx1 + sx2 * bw
    # nby2 = nby1 + sy2 * bh

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

def get_patches_tensor(img):
    imgpatches = get_patches(img, patch_size=testconf.crop_size, patch_num=testconf.crop_num)
    augment_patches = torch.FloatTensor(testconf.crop_num, 3, testconf.crop_size, testconf.crop_size).to(device)

    for i in range(testconf.crop_num):
        onepatch = imgpatches[i]
        onepatch = onepatch.astype(np.float32)
        onepatch = onepatch / 255.0
        onepatch = onepatch.transpose(2, 0, 1)
        onepatch = torch.from_numpy(onepatch).to(device)
        augment_patches[i, :, :, :] = onepatch
    return augment_patches

def get_patches_better(img):
    imgpatches, nump = get_patches_augment(img, patch_size=testconf.crop_size, timenum=testconf.crop_scale)

    if nump == 0:
        augment_patches =[]
        return augment_patches

    augment_patches = torch.FloatTensor(nump, 3, testconf.crop_size, testconf.crop_size).to(device)

    for i in range(nump):
        onepatch = imgpatches[i]
        onepatch = onepatch.astype(np.float32)
        onepatch = onepatch / 255.0
        onepatch = onepatch.transpose(2, 0, 1)
        onepatch = torch.from_numpy(onepatch).to(device)
        augment_patches[i, :, :, :] = onepatch
    return augment_patches

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
    test_patches = get_patches_tensor(img_mat)

    out = snet(test_patches)
    out = torch.sigmoid(out)
    len_out = out.shape[0]
    max_score = torch.max(out)
    min_score = torch.min(out)
    blur_score = (torch.sum(out) - max_score - min_score) / (len_out - 2)
    # blur_score = torch.mean(out)
    showscore = np.around(blur_score.item(), 4)

    posx = int(5)
    posy = int(5)
    cv2.putText(img_mat, str(showscore), (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 255), 4)
    cv2.rectangle(img_mat, (0, 0), (im_w, im_h), (0, 255, 0), 4)

    if dir:
        return img_mat, showscore
    else:
        cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        cv2.imshow('result', img_mat)
        cv2.waitKey(0)

def test_patch_nodet(img_path, snet, dir=False):
    img_mat = cv2.imread(img_path, cv2.IMREAD_COLOR)
    im_h, im_w, _ = img_mat.shape
    test_patches = get_patches_better(img_mat)

    out = snet(test_patches)
    out = torch.sigmoid(out)
    len_out = out.shape[0]
    max_score = torch.max(out)
    min_score = torch.min(out)
    blur_score = (torch.sum(out) - max_score - min_score) / (len_out - 2)

    # blur_score = torch.mean(out)
    showscore = np.around(blur_score.item(), 4)

    posx = int(5)
    posy = int(5)
    cv2.putText(img_mat, str(showscore), (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 255), 4)
    cv2.rectangle(img_mat, (0, 0), (im_w, im_h), (0, 255, 0), 4)

    if dir:
        return img_mat, showscore
    else:
        cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        cv2.imshow('result', img_mat)
        cv2.waitKey(0)

def test_one(img_path, dnet, snet, minface, dir=False):
    img_mat = cv2.imread(img_path, cv2.IMREAD_COLOR)
    im_h, im_w, _ = img_mat.shape
    face_rect, key_points = detect_one_img(dnet, img_mat, minface)
    showscore = 0.0
    for box, lands in zip(face_rect, key_points):
        new_box = expand_facebox(box, im_w, im_h)#人脸框四周扩充
        # new_box = np.zeros(4, dtype=np.int32)
        # new_box[0] = 0
        # new_box[1] = 0
        # new_box[2] = im_w
        # new_box[3] = im_h

        face_roi = img_mat[new_box[1]:new_box[3], new_box[0]:new_box[2], :]

        # test_patches = get_patches_tensor(face_roi)
        test_patches = get_patches_better(face_roi)
        if len(test_patches) == 0:
            showscore = 0.0
            return img_mat, showscore
        # b, c, h, w = roi_process.shape
        # save_feature_channel("txt/imgp.txt", roi_process, b, c, h, w)

        out = snet(test_patches)
        out = torch.sigmoid(out)
        blur_score = torch.mean(out)
        showscore = np.around(blur_score.item(), 4)

        posx = int(new_box[0])
        posy = int(new_box[1])
        cv2.putText(img_mat, str(showscore), (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 255), 4)
        cv2.rectangle(img_mat, (new_box[0], new_box[1]), (new_box[2], new_box[3]), (0, 255, 0), 4)

    if dir:
        return img_mat, showscore
    else:
        cv2.namedWindow('result', cv2.WINDOW_NORMAL)
        cv2.imshow('result', img_mat)
        cv2.waitKey(0)

def test_dir(imdir, savedir, net1, net2, min_face=60):
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    filetxt = open("D:/data/imgs/facePicture/blur/test/result_2.txt", "w+")
    for root, dirs, files in os.walk(imdir):
        for file in files:
            # filetxt.write(file + ":     ")
            root = root.replace('\\', '/')
            imgpath = root + "/" + file
            savepath = savedir + "/" + file
            saveimg, _score = test_one(imgpath, net1, net2, min_face, dir=True)
            # saveimg, _score = test_one_nodet(imgpath, net2, dir=True)
            # saveimg, _score = test_patch_nodet(imgpath, net2, dir=True)
            _score = str(_score)
            filetxt.write(_score + "\n")
            cv2.imshow('result', saveimg)
            cv2.waitKey(1)
    filetxt.close()

def test_rename_dir(imdir, net1, net2, min_face=60):
    for root, dirs, files in os.walk(imdir):
        for file in files:
            mohu = 0
            imgname, imghz = file.split(".")
            imgpath = imdir + "/" + file
            # savepath = "D:/wx/aa" + "/" + file
            # shutil.move(imgpath, savepath)
            saveimg, _score = test_one(imgpath, net1, net2, min_face, dir=True)
            if _score > 0.5:
                mohu = 1
            savename = imdir + "/" + imgname + "_" + str(mohu) + "." + imghz
            os.rename(imgpath, savename)

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

def crop_FacePatches_dir(imgdir, savedir, patchSize, patchNum):
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            imgname, houzui = file.split(".")
            imgpath = root + "/" + file
            dirpath = savedir + "/" + imgname
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            img_mat = cv2.imread(imgpath, cv2.IMREAD_COLOR)
            patches = get_patches(img_mat, patch_size=patchSize, patch_num=patchNum)
            for i in range(patchNum):
                patchone = patches[i]
                savepath = dirpath + "/" + str(i) + file
                cv2.imwrite(savepath, patchone)

def get_score_byname(imgdirs, txtsave):
    label_classfication = open(txtsave, mode="w+")
    for root, dirs, files in os.walk(imgdirs):
        for file in files:
            splitfile = file.split(".")[0]
            namesplit = splitfile.split("_")
            lab = int(namesplit[-1])
            change_lab = 0.0
            if lab == 0:
                change_lab = 0.0
            if lab == 1:
                change_lab = 0.25
            if lab == 2:
                change_lab = 0.5
            if lab == 3:
                change_lab = 0.75
            if lab == 4:
                change_lab = 1.0
            label_classfication.write(str(change_lab) + "\n")
    label_classfication.close()

def get_predict_result(imdir, net1, net2, txt1, txt2, txtlab):
    txt1 = open(txt1, "w+")
    txt2 = open(txt2, "w+")
    txtlab = open(txtlab, "w+")
    for root, dirs, files in os.walk(imdir):
        for file in files:
            splitfile = file.split(".")[0]
            namesplit = splitfile.split("_")
            lab = int(namesplit[-1])
            change_lab = 0.0
            if lab == 0:
                change_lab = 0.0
            if lab == 1:
                change_lab = 0.25
            if lab == 2:
                change_lab = 0.5
            if lab == 3:
                change_lab = 0.75
            if lab == 4:
                change_lab = 1.0
            txtlab.write(str(change_lab) + "\n")

            root = root.replace('\\', '/')
            imgpath = root + "/" + file
            saveimg1, _score1 = test_one_nodet(imgpath, net1, dir=True)
            saveimg2, _score2 = test_patch_nodet(imgpath, net2, dir=True)
            _score1 = str(_score1)
            _score2 = str(_score2)
            txt1.write(_score1 + "\n")
            txt2.write(_score2 + "\n")
    txt1.close()
    txt2.close()
    txtlab.close()

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

def gaussion_blur(imdir, savedir):
    for root, dirs, files in os.walk(imdir):
        for file in files:
            root = root.replace('\\', '/')
            imgname, hz = file.split(".")
            imgpath = root + "/" + file
            # savep = savedir + "/" + file
            # savep1 = savedir + "/" + imgname + "_1." + hz
            # savep2 = savedir + "/" + imgname + "_2." + hz
            # savep3 = savedir + "/" + imgname + "_3." + hz
            # savep4 = savedir + "/" + imgname + "_4." + hz
            img_mat = cv2.imread(imgpath, cv2.IMREAD_COLOR)
            savep = savedir + "/5/" + imgname + "_4." + hz
            cv2.imwrite(savep, img_mat)

            # rand_type = random.choice([0, 1, 2, 3, 4])
            # if rand_type == 0:
            #     savep = savedir + "/0/" + imgname + "_0." + hz
            #     cv2.imwrite(savep, img_mat)
            # if rand_type == 1:
            #     blur = cv2.GaussianBlur(img_mat, (11, 11), 0.8)
            #     # blur = cv2.GaussianBlur(img_mat, (5, 5), 0.6)
            #     savep = savedir + "/1/" + imgname + "_1." + hz
            #     cv2.imwrite(savep, blur)
            # if rand_type == 2:
            #     blur = cv2.GaussianBlur(img_mat, (13, 13), 1.3)
            #     # blur = cv2.GaussianBlur(img_mat, (7, 7), 1.0)
            #     savep = savedir + "/2/" + imgname + "_2." + hz
            #     cv2.imwrite(savep, blur)
            # if rand_type == 3:
            #     blur = cv2.GaussianBlur(img_mat, (15, 15), 1.8)
            #     # blur = cv2.GaussianBlur(img_mat, (9, 9), 1.4)
            #     savep = savedir + "/3/" + imgname + "_3." + hz
            #     cv2.imwrite(savep, blur)
            # if rand_type == 4:
            #     blur = cv2.GaussianBlur(img_mat, (17, 17), 2.2)
            #     # blur = cv2.GaussianBlur(img_mat, (11, 11), 1.7)
            #     savep = savedir + "/4/" + imgname + "_4." + hz
            #     cv2.imwrite(savep, blur)

            # blur1 = cv2.GaussianBlur(img_mat, (5, 5), 0.6)
            # blur2 = cv2.GaussianBlur(img_mat, (7, 7), 1.0)
            # blur3 = cv2.GaussianBlur(img_mat, (9, 9), 1.4)
            # blur4 = cv2.GaussianBlur(img_mat, (11, 11), 1.8)
            # cv2.imwrite(savep, img_mat)
            # cv2.imwrite(savep1, blur1)
            # cv2.imwrite(savep2, blur2)
            # cv2.imwrite(savep3, blur3)
            # cv2.imwrite(savep4, blur4)
            # print("done")

if __name__ == "__main__":
    qnet = FaceQualityNet(channels=testconf.net_channels, lda_outc=testconf.lad_channel)  # 需要修改
    q_path = "weights/FaceQuality.pth"  # 需要修改
    # qnet = FaceQualitySlim(channels=testconf.slim_channels)
    # q_path = "weights/FaceQualitySlim_500.pth"
    q_dict = torch.load(q_path, map_location=lambda storage, loc: storage)
    qnet.load_state_dict(q_dict)
    qnet.eval()
    qnet = qnet.to(device)

    qnet2 = FaceQualityNet(channels=testconf.net_channels, lda_outc=testconf.lad_channel)
    q_path2 = "weights/FaceQuality_20200109.pth"
    q_dict2 = torch.load(q_path2, map_location=lambda storage, loc: storage)
    qnet2.load_state_dict(q_dict2)
    qnet2.eval()
    qnet2 = qnet2.to(device)
    # saveparams = pytorch_to_dpcoreParams(qnet2)
    # saveparams.forward("FaceQuality_param_cfg.h", "FaceQuality_param_src.h")

    dnet = FaceDetectSlimNet(cfg=cfg)  # 需要修改
    d_path = "weights/face_slim_0609_250.pth"  # 需要修改
    d_dict = torch.load(d_path, map_location=lambda storage, loc: storage)
    dnet.load_state_dict(d_dict)
    dnet.eval()
    dnet = dnet.to(device)

    imgpath = "D:/data/imgs/facePicture/blur/test/1/10078.jpg"
    savepath = "result/res.jpg"
    min_face = 60
    # test_one(imgpath, dnet, qnet2, min_face, dir=False)

    imgdir = "D:/wx/1117"
    savedir = "D:/data/imgs/facePicture/blur/faces/add"
    txt1 = "D:/data/imgs/facePicture/blur/test/result_1.txt"
    txt2 = "D:/data/imgs/facePicture/blur/test/result_2.txt"
    txtl = "D:/data/imgs/facePicture/blur/test/label.txt"
    # test_dir(imgdir, savedir, dnet, qnet, min_face)
    test_rename_dir(imgdir, dnet, qnet, min_face)
    # get_face_dirs(imgdir, savedir, dnet)
    # get_score_byname(imgdir, txt)
    # get_predict_result(imgdir, qnet, qnet2, txt1, txt2, txtl)

    imgd = "D:/data/imgs/facePicture/blur/select/4"
    saved = "D:/data/imgs/facePicture/blur/patches"
    # crop_FacePatches_dir(imgd, saved, 96, 16)
    # create_train_samples(imgd, saved, dnet, qnet2)
    # gaussion_blur(imgd, saved)
    print("done")











