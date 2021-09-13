import torch
import cv2
import os
import numpy as np
from huoti_net import Huoti
from myconfig import config as testconf
from PIL import Image, ImageDraw, ImageFont

from detector.config import cfg_mnet as cfg
from detector.create_anchors import PriorBox
from detector.mobilev3_face import mobilev3Fpn_small
from detector.retinaface_utils import decode, decode_landm
from detector.nms import py_cpu_nms
device = "cpu"


def paint_chinese_opencv(im, chinese, pos, wordsize, color):
    img_PIL = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('SIMYOU.TTF', wordsize)
    fillColor = color  # (255,0,0)
    position = pos  # (100,100)
    # chinese = chinese.decode('utf-8')
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, chinese, font=font, fill=fillColor)

    img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img

def expand_facebox(rect, imgw, imgh):
    bx = rect[0]
    by = rect[1]
    bw = rect[2] - rect[0]
    bh = rect[3] - rect[1]

    # face
    nbx1 = bx - 0.1 * bw
    nby1 = by - 0.1 * bh
    nbx2 = nbx1 + 1.2 * bw
    nby2 = nby1 + 1.1 * bh

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
    im = im - testconf.rgb_mean
    # im = im / 255.0
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
        # roi_process = img_process(face_roi)  # 数据处理，转为网络输入的形式

        roi_process = cv2.resize(face_roi, (128, 128), interpolation=cv2.INTER_CUBIC)
        roi_process = roi_process.astype(np.float32)
        roi_process -= testconf.rgb_mean
        roi_process = roi_process.transpose((2, 0, 1))
        roi_process = torch.from_numpy(roi_process)
        roi_process = roi_process.unsqueeze(0)

        # b, c, h, w = roi_process.shape
        # save_feature_channel("txt/imgp.txt", roi_process, b, c, h, w)

        outputs = snet(roi_process)
        _, prediction = torch.max(outputs.data, 1)
        posx = int(0.35 * new_box[0] + 0.35 * new_box[2])
        posy = int(new_box[1] + 10)
        if prediction == 0:
            img_mat = paint_chinese_opencv(img_mat, "非活体", (posx, posy), 40, (255, 0, 0))
            # cv2.putText(img_mat, "fake", (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 255), 4)
        if prediction == 1:
            img_mat = paint_chinese_opencv(img_mat, "活体", (posx, posy), 40, (255, 0, 0))
            # cv2.putText(img_mat, "true", (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 255), 4)

        cv2.rectangle(img_mat, (new_box[0], new_box[1]), (new_box[2], new_box[3]), (0, 255, 0), 4)

    if dir:
        return img_mat, prediction
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
            saveimg, _preval = test_one(imgpath, net1, net2, min_face, dir=True)
            cv2.imshow('result', saveimg)
            cv2.waitKey(1)
            if _preval == 0:
                cv2.imwrite(savepath, saveimg)


if __name__ == "__main__":

    # rgb_mean = (104, 117, 123)
    #
    # myNet = Huoti(nclasses=2)
    # myNet.eval()
    #
    # # # GPU
    # # myNet = myNet.to(torch.device("cuda"))
    #
    # ht_model = torch.load("weights/ht_model_20200330.pth")
    # myNet.load_state_dict(ht_model)
    # images = cv2.imread("imgs/untrue_real(2066).jpg")
    # images = cv2.resize(images, (128, 128), interpolation=cv2.INTER_CUBIC)
    # images = images.astype(np.float32)
    # images -= rgb_mean
    # images = images.transpose((2, 0, 1))
    # image = torch.from_numpy(images)
    # image = image.unsqueeze(0)
    #
    # # # GPU
    # # image = image.to(torch.device("cuda"))
    #
    # outputs = myNet(image)
    # _, prediction = torch.max(outputs.data, 1)
    # if prediction == 0:
    #     print("非活体")
    # if prediction == 1:
    #     print("活体")



    hnet = Huoti(nclasses = 2)  # 需要修改
    h_path = "weights/ht_model_20200330.pth"  # 需要修改
    h_dict = torch.load(h_path, map_location=lambda storage, loc: storage)
    hnet.load_state_dict(h_dict)
    hnet.eval()
    hnet = hnet.to(device)
    # saveparams = pytorch_to_dpcoreParams(snet)
    # saveparams.forward("NeckShadow_param_cfg.h", "NeckShadow_param_src.h")

    dnet = mobilev3Fpn_small(cfg=cfg)  # 需要修改
    d_path = "detector/mobilev3Fpn_0810_250.pth"  # 需要修改
    d_dict = torch.load(d_path, map_location=lambda storage, loc: storage)
    dnet.load_state_dict(d_dict)
    dnet.eval()
    dnet = dnet.to(device)

    imgpath = "D:/huoti_detect/temp/f1e30d98f111ea9dc100163e0070b6.jpg"
    savepath = "result/res.jpg"
    min_face = 60
    test_one(imgpath, dnet, hnet, min_face, dir=False)

    imgdir = "D:/data/imgs/facePicture/shadow/test/with"
    savedir = "D:/data/imgs/facePicture/shadow/test/error"
    # test_dir(imgdir, savedir, dnet, snet, min_face)
    # get_face_dirs(imgdir, savedir, dnet)
    print("done")











