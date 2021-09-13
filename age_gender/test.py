import torch
import math
import numpy as np
import os
import cv2
import imutils
import random
from fangshe_change import get_fangsheMatrix, REFERENCE_FACIAL_POINTS

from mobilev3_small import mobilev3_AgeGenderNet, mobilev3_AGDLDL, mobilev3_AGDLDL_new
from myconfig import config as testconf

from retinaface.create_anchors import PriorBox
from retinaface.config import cfg_slimNet3 as cfg
from retinaface.face_net import FaceDetectSlimNet
from retinaface.retinaface_utils import decode, decode_landm
from retinaface.nms import py_cpu_nms

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

    nbx1 = bx - 0.15 * bw
    nby1 = by - 0.05 * bh
    nbx2 = nbx1 + 1.3 * bw
    nby2 = nby1 + 1.1 * bh

    # nbx1 = bx - 0.1 * bw
    # nby1 = by - 0.04 * bh
    # nbx2 = nbx1 + 1.2 * bw
    # nby2 = nby1 + 1.08 * bh

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
    im = cv2.resize(img, (testconf.img_width, testconf.img_height), interpolation=cv2.INTER_CUBIC)
    im = im.astype(np.float32)
    im = (im - testconf.bgr_mean) / testconf.bgr_std
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

def test_one(img_mat, dnet, agnet, minface):
    # img_mat = cv2.imread(imgp, cv2.IMREAD_COLOR)
    im_h, im_w, _ = img_mat.shape
    face_rect, key_points = detect_one_img(dnet, img_mat, minface)
    for box, lands in zip(face_rect, key_points):
        bx1 = box[0]
        by1 = box[1]
        bx2 = box[2]
        by2 = box[3]

        landxy = lands.reshape(-1, 2)
        similar_trans_matrix = get_fangsheMatrix(landxy, POINTS)

        aligned_face = cv2.warpAffine(img_mat.copy(), similar_trans_matrix, (128, 128))
        # cv2.namedWindow('fs', cv2.WINDOW_NORMAL)
        # cv2.imshow('fs', aligned_face)
        # cv2.waitKey(0)

        new_box = expand_facebox(box, im_w, im_h)#人脸框四周扩充
        face_roi = img_mat[new_box[1]:new_box[3], new_box[0]:new_box[2], :]
        # roi_process = img_process(face_roi)#数据处理，转为网络输入的形式
        roi_process = img_process(aligned_face)  # 数据处理，转为网络输入的形式
        page, pgender = agnet(roi_process)
        _, m_wm = torch.max(pgender.data, 1)
        age_num = page.data.squeeze(0)
        age_num = age_num.cpu().numpy()
        age_num = int(age_num[0] * 116.0)
        posx = int(box[0])
        posy = int(box[1])
        mantxt = "man:" + str(age_num)
        womantxt = "lady:" + str(age_num)
        if m_wm == 0:
            cv2.putText(img_mat, mantxt, (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 255), 6)
        if m_wm == 1:
            cv2.putText(img_mat, womantxt, (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 255), 6)

        cv2.rectangle(img_mat, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        pp = np.zeros(8, dtype=np.int32)
        pp[0] = lands[0]
        pp[1] = lands[1]
        pp[2] = lands[2]
        pp[3] = lands[3]
        pp[4] = lands[6]
        pp[5] = lands[7]
        pp[6] = lands[8]
        pp[7] = lands[9]
        cv2.circle(img_mat, (lands[0], lands[1]), 1, (0, 0, 255), 4)
        cv2.circle(img_mat, (lands[2], lands[3]), 1, (0, 255, 255), 4)
        cv2.circle(img_mat, (lands[4], lands[5]), 1, (255, 0, 255), 4)
        cv2.circle(img_mat, (lands[6], lands[7]), 1, (0, 255, 0), 4)
        cv2.circle(img_mat, (lands[8], lands[9]), 1, (255, 0, 0), 4)
    return img_mat, age_num, m_wm

def test_DLDL(img_mat, dnet, agnet, minface):
    # img_mat = cv2.imread(imgp, cv2.IMREAD_COLOR)
    im_h, im_w, _ = img_mat.shape
    face_rect, key_points = detect_one_img(dnet, img_mat, minface)
    rank = torch.Tensor([i for i in range(116)]).cuda()
    gd = 1
    ag = 20
    for box, lands in zip(face_rect, key_points):
        bx1 = box[0]
        by1 = box[1]
        bx2 = box[2]
        by2 = box[3]

        # landxy = lands.reshape(-1, 2)
        # similar_trans_matrix = get_fangsheMatrix(landxy, POINTS)
        # aligned_face = cv2.warpAffine(img_mat.copy(), similar_trans_matrix, (144, 144))

        new_box = expand_facebox(box, im_w, im_h)  # 人脸框四周扩充
        face_roi = img_mat[new_box[1]:new_box[3], new_box[0]:new_box[2], :]
        roi_process = img_process(face_roi)  # 数据处理，转为网络输入的形式
        page, pgender = agnet(roi_process)
        _, m_wm = torch.max(pgender.data, 1)
        predict_age = torch.sum(page * rank, dim=1).item()
        age_num = int(predict_age)
        gd = m_wm
        ag = age_num
        posx = int(box[0])
        posy = int(box[1])
        mantxt = "man:" + str(age_num)
        womantxt = "man:" + str(age_num)
        if m_wm == 0:
            cv2.putText(img_mat, mantxt, (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 8.0, (0, 0, 255), 8)
        if m_wm == 1:
            cv2.putText(img_mat, womantxt, (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 8.0, (0, 0, 255), 8)

        # cv2.rectangle(img_mat, (bx1, by1), (bx2, by2), (0, 255, 0), 8)
        cv2.rectangle(img_mat, (new_box[0], new_box[1]), (new_box[2], new_box[3]), (0, 255, 0), 8)
        pp = np.zeros(10, dtype=np.int32)
        pp[0] = lands[0]
        pp[1] = lands[1]
        pp[2] = lands[2]
        pp[3] = lands[3]
        pp[4] = lands[4]
        pp[5] = lands[5]
        pp[6] = lands[6]
        pp[7] = lands[7]
        pp[8] = lands[8]
        pp[9] = lands[9]
        cv2.circle(img_mat, (pp[0], pp[1]), 1, (0, 0, 255), 4)
        cv2.circle(img_mat, (pp[2], pp[3]), 1, (0, 255, 255), 4)
        cv2.circle(img_mat, (pp[4], pp[5]), 1, (255, 0, 255), 4)
        cv2.circle(img_mat, (pp[6], pp[7]), 1, (0, 255, 0), 4)
        cv2.circle(img_mat, (pp[8], pp[9]), 1, (255, 0, 0), 4)
    return img_mat, ag, gd

def test_alignDLDL(img_mat, agnet):
    rank = torch.Tensor([i for i in range(116)]).cuda()
    roi_process = img_process(img_mat)  # 数据处理，转为网络输入的形式
    page, pgender = agnet(roi_process)
    _, m_wm = torch.max(pgender.data, 1)
    predict_age = torch.sum(page * rank, dim=1).item()
    age_num = int(predict_age)
    posx = 0
    posy = 32
    mantxt = "man:" + str(age_num)
    womantxt = "lady:" + str(age_num)
    if m_wm == 0:
        cv2.putText(img_mat, mantxt, (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 4)
    if m_wm == 1:
        cv2.putText(img_mat, womantxt, (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 4)
    return img_mat

def test_dir(imdir, savedir, net1, net2, min_face):
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    for root, dirs, files in os.walk(imdir):
        for file in files:
            if file.endswith("png"):
                root = root.replace('\\', '/')
                imgpath = root + "/" + file
                imgdir, imgname = os.path.split(imgpath)
                savepath = os.path.join(savedir, imgname)
                img_mat = cv2.imread(imgpath, cv2.IMREAD_COLOR)
                saveimg, age, gender = test_DLDL(img_mat, net1, net2, min_face)
                cv2.imshow('result', saveimg)
                cv2.waitKey(1)
                cv2.imwrite(savepath, saveimg)

def str_in_lists(strr, wlist=[]):
    if strr in wlist:
        return 1
    else:
        return 0

def test_rename_dir(imdir, net1, net2, min_face):
    for root, dirs, files in os.walk(imdir):
        for file in files:
            xinbie = 0
            imgpath = imdir + "/" + file
            imgname, imghz = file.split(".")
            img_mat = cv2.imread(imgpath)
            if img_mat is not None:
                saveimg, age, gender = test_DLDL(img_mat, net1, net2, min_face)
                if gender == 0:
                    xinbie = 1
                if gender == 1:
                    xinbie = 0
                savename = imdir + "/" + imgname + "_" + str(xinbie) + "." + imghz
                os.rename(imgpath, savename)
            else:
                print(imgpath)

def test_alignface_imgs(imdir, saveerror, net):
    rank = torch.Tensor([i for i in range(116)]).cuda()
    total_imgNum = 0
    right_num = 0
    for root, dirs, files in os.walk(imdir):
        for file in files:
            if file.endswith("jpg"):
                total_imgNum += 1
                root = root.replace('\\', '/')
                rootsplit = root.split("/")  # 获取性别
                gender_label = int(rootsplit[-1])
                imgpath = root + "/" + file
                img_mat = cv2.imread(imgpath, cv2.IMREAD_COLOR)
                if img_mat is not None:
                    roi_process = img_process(img_mat)  # 数据处理，转为网络输入的形式
                    page, pgender = net(roi_process)
                    _, m_wm = torch.max(pgender.data, 1)
                    predict_age = torch.sum(page * rank, dim=1).item()
                    age_num = int(predict_age)
                    if (m_wm == gender_label):
                        right_num += 1
                    else:
                        posx = 40
                        posy = 48
                        mantxt = "man:" + str(age_num)
                        womantxt = "lady:" + str(age_num)
                        if m_wm == 0:
                            cv2.putText(img_mat, mantxt, (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 4)
                        if m_wm == 1:
                            cv2.putText(img_mat, womantxt, (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 4)
                        savepath = saveerror + "/" + rootsplit[-1] + "/" + file
                        cv2.imwrite(savepath, img_mat)
    true_rate = float(right_num / total_imgNum)
    print("gender detect rate: {:.4f}".format(true_rate))


def value_gender(imdir, savedir, net1, net2, min_face):
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    total_imgNum = 0
    right_num = 0
    for root, dirs, files in os.walk(imdir):
        for file in files:
            total_imgNum += 1
            if file.endswith("jpg"):
                root = root.replace('\\', '/')
                rootsplit = root.split("/")  # 获取性别
                gender_label = int(rootsplit[-1])
                imgpath = root + "/" + file
                img_mat = cv2.imread(imgpath, cv2.IMREAD_COLOR)
                saveimg, age, gender = test_DLDL(img_mat, net1, net2, min_face)
                if(gender == gender_label):
                    right_num += 1
                else:
                    savepath = savedir + "/" + rootsplit[-1] + "/" + file
                    cv2.imwrite(savepath, saveimg)
                cv2.imshow('result', saveimg)
                cv2.waitKey(1)
    true_rate = float(right_num / total_imgNum)
    print("gender detect rate: {:.4f}".format(true_rate))


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
            res_frame, age, gender = test_DLDL(frame, net1, net2, min_face)
            outvideo.write(res_frame)
            cv2.imshow('result1', res_frame)
            cv2.waitKey(10)
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

    # 性别年龄
    AGnet = mobilev3_AGDLDL_new()  # 需要修改
    pretrained_ag = "weights/AgeGender_0713.pth"  # 需要修改
    ag_dict = torch.load(pretrained_ag, map_location=lambda storage, loc: storage)
    if "state_dict" in ag_dict.keys():
        ag_dict = remove_prefix(ag_dict['state_dict'], 'module.')
    else:
        ag_dict = remove_prefix(ag_dict, 'module.')
    check_keys(AGnet, ag_dict)
    AGnet.load_state_dict(ag_dict, strict=False)
    AGnet.eval()
    print('Finished loading model!')
    AGnet = AGnet.to(device)

    imgpath = "D:/data/imgs/facePicture/blur/test/1/10078.jpg"
    savepath = "D:/codes/project/ageTest/result/af93c266db7b11eaad2600163e0070b6.jpg"
    min_face = 150


    # im = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    # simg, age, gender = test_DLDL(im, dnet, AGnet, min_face)
    # # simg = test_alignDLDL(im, AGnet)
    # cv2.namedWindow('result1', cv2.WINDOW_NORMAL)
    # cv2.imshow('result1', simg)
    # cv2.waitKey(0)

    videop = ""
    savep = "result/1.mp4"
    # test_camORvideo(videop, savep, dnet, AGnet, min_face)

    dir = "D:/wx/1117"
    save = "D:/codes/project/ageTest/result"
    # test_dir(dir, save, dnet, AGnet, min_face)
    test_rename_dir(dir, dnet, AGnet, min_face)
    # value_gender(dir, save, Dnet, AGnet, min_face)

    dirimg1 = "D:/data/imgs/facePicture/Age_Gender/IDPhoto/result"
    imgsave1 = "D:/data/imgs/facePicture/Age_Gender/IDPhoto/error"
    # test_alignface_imgs(dirimg1, imgsave1, AGnet)










