import os
import cv2
import numpy as np
import torch
import time

from detector.config import cfg_mnet
from detector.create_anchors import PriorBox
from detector.mobilev3_face import mobilev3Fpn_small
from detector.retinaface_utils import decode, decode_landm
from detector.nms import py_cpu_nms

cfg = cfg_mnet   #需要修改
rgb_mean = cfg['rgb_mean'] # bgr order
std_mean = cfg['std_mean']
img_size = cfg["image_size"]
conf_thresh = 0.5
nms_thresh = 0.35
device = torch.device("cpu")

net = mobilev3Fpn_small(cfg=cfg)   #需要修改
pretrained_path = "detector/mobilev3Fpn_0810_250.pth"   #需要修改
pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
net.load_state_dict(pretrained_dict, strict=False)
net = net.to(device)
net.eval()
print('Finished loading face detect weights!')

def net_forward(faceNet, img_data, minface):
    img = np.float32(img_data)
    im_shape = img.shape
    im_size_max = np.max(im_shape[0:2])
    res_scal = 20 / float(minface)
    # res_scal = 420.0 / im_size_max
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

    tic = time.time()
    loc, conf, landms = faceNet(img)  # forward pass
    print('640 net forward time: {:.4f}'.format(time.time() - tic))
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


def get_channel_mean(imgbgr, mode=cv2.COLOR_BGR2Lab):
    # p = np.zeros([1, 1, 3], np.float32)
    # p[0, 0, 0] = 173
    # p[0, 0, 1] = 120
    # p[0, 0, 2] = 57
    # pp = p / 255.0
    # ppp = cv2.cvtColor(pp, cv2.COLOR_BGR2LAB)
    # meanl = np.mean(ppp[:, :, 0])
    # meana = np.mean(ppp[:, :, 1])
    # meanb = np.mean(ppp[:, :, 2])



    cv2.imshow('imgbgr', imgbgr)
    imggy = imgbgr.astype(np.float32) / 255.0
    imglab = cv2.cvtColor(imggy, mode)
    cv2.imshow('imglab', imglab)
    cv2.waitKey(1)

    if mode == cv2.COLOR_BGR2Lab:
        l_mean = np.mean(imglab[:, :, 0])
        a_mean = np.mean(imglab[:, :, 1])
        b_mean = np.mean(imglab[:, :, 2])
        return l_mean, a_mean, b_mean
    if mode == cv2.COLOR_BGR2RGB:
        b_mean = np.mean(imgbgr[:, :, 0])
        g_mean = np.mean(imgbgr[:, :, 1])
        r_mean = np.mean(imgbgr[:, :, 2])
        return r_mean, g_mean, b_mean


def face_mean(imgp, dnet, minface):
    img_mat = cv2.imread(imgp, cv2.IMREAD_COLOR)
    im_h, im_w, _ = img_mat.shape
    face_rect, key_points = net_forward(dnet, img_mat, minface)
    for box, lands in zip(face_rect, key_points):
        bx1 = 116
        by1 = 114
        bx2 = 242
        by2 = 251
        # bx1 = int(box[0])
        # by1 = int(box[1])
        # bx2 = int(box[2])
        # by2 = int(box[3])
        face_roi = img_mat[by1:by2, bx1:bx2, :]
        lmean, amean, bmean = get_channel_mean(face_roi, mode=cv2.COLOR_BGR2RGB)
        print(lmean, amean, bmean)

def face_mean_mask(imgp, maskp):
    img_mat = cv2.imread(imgp, cv2.IMREAD_COLOR)
    mask_mat = cv2.imread(maskp, cv2.IMREAD_COLOR)
    face_mask = (mask_mat == 255).astype(np.float32)
    h_face, w_face = np.nonzero(face_mask[:, :, 1])
    lenface = w_face.size
    meanb = 0
    meang = 0
    meanr = 0
    for i in range(lenface):
        meanb += img_mat[h_face[i], w_face[i], 0]
        meang += img_mat[h_face[i], w_face[i], 1]
        meanr += img_mat[h_face[i], w_face[i], 2]

    meanb /= (1.0 * lenface)
    meang /= (1.0 * lenface)
    meanr /= (1.0 * lenface)
    return meanb, meang, meanr
    # print(meanb, meang, meanr)

def get_lab_mean(imgp, maskp):
    img_ori = cv2.imread(imgp)
    img_seg = cv2.imread(maskp)
    # 2、获取分割图中的面部区域
    face_mask = (img_seg == 255).astype(np.float32)
    # 求面部区域的和，因为值全为1，所以只需求值为1的长度
    rows_face, cols_face = np.nonzero(face_mask[:, :, 1])
    sum_face = len(rows_face)
    # 3、将face_mask与原始图片点乘
    img_src = img_ori.astype(np.float32)
    img_fin = np.multiply(img_src, face_mask) / 255.

    # 4、将BGR转换成LAB
    img_lab = cv2.cvtColor(img_fin, cv2.COLOR_RGB2Lab)
    # cv2.namedWindow('result1', cv2.WINDOW_NORMAL)
    #     # cv2.imshow('result1', img_lab)
    #     # cv2.waitKey(0)

    # 5.求L通道的均值，判断亮度
    l_channel, a_channel, b_channel = cv2.split(img_lab)
    # 求出L,a,b通道的非零值的位置，然后对这些位置进行求和，随后再计算均值
    rows_l, cols_l = np.nonzero(l_channel)
    sum_l = sum(l_channel[rows_l, cols_l])
    mean_l = (sum_l / sum_face)

    rows_a, cols_a = np.nonzero(a_channel)
    sum_a = sum(a_channel[rows_a, cols_a])
    mean_a = (sum_a / sum_face)

    rows_b, cols_b = np.nonzero(b_channel)
    sum_b = sum(b_channel[rows_b, cols_b])
    mean_b = (sum_b / sum_face)
    return mean_l, mean_a, mean_b

def get_labmean_dir(imgdir, maskdir):
    for root, dirs, files in os.walk(imgdir):
        imgnum = len(files)
        for file in files:
            if file.endswith("jpg"):
                splitfile = file.split(".jpg")
                imgname = splitfile[0]
                imgpath = imgdir + "/" + imgname + ".jpg"
                maskpath = maskdir + "/" + imgname + "m.jpg"
                val_l, val_a, val_b = get_lab_mean(imgpath, maskpath)
                l = np.round(val_l, 3)
                a = np.round(val_a, 3)
                b = np.round(val_b, 3)

                print(imgpath, ": ", l, a, b)

if __name__ == "__main__":
    imgp = "test/IMG_0048.jpg"
    maskp = "test/IMG_0048m.jpg"
    # lmeanv = get_lab_mean(imgp, maskp)

    imgd = "D:/data/imgs/facePicture/face_bright/face_skin/norm"
    maskd = "D:/data/imgs/facePicture/face_bright/face_skin/mask"
    # get_labmean_dir(imgd, maskd)

    # img = cv2.imread(imgp, cv2.IMREAD_COLOR)

    # lm, am, bm = get_channel_mean(img, mode=cv2.COLOR_BGR2RGB)
    # print(lm, am, bm)
    # face_mean(imgp, net, 60)

    bm, gm, rm = face_mean_mask(imgp, maskp)
    print(bm, gm, rm)







