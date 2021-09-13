import torch
import math
import numpy as np
import os
from retinaface import prior_box
import cv2
import imutils
from retinaface.net_rfb import RFB
from retinaface.box_utils import decode, decode_landm
from retinaface.py_cpu_nms import py_cpu_nms
from slim_net import maskNet
from test_config import config as testconf
import random

device = "cuda"
cfg_rfb = testconf.rfb

def get_angle(kpoints):
    lt_x = kpoints[0]
    lt_y = kpoints[1]
    rt_x = kpoints[2]
    rt_y = kpoints[3]
    lb_x = kpoints[4]
    lb_y = kpoints[5]
    rb_x = kpoints[6]
    rb_y = kpoints[7]

    mean_tx = 0.5 * (lt_x + rt_x)
    mean_ty = 0.5 * (lt_y + rt_y)
    mean_bx = 0.5 * (lb_x + rb_x)
    mean_by = 0.5 * (lb_y + rb_y)
    k_lt_lb = 1111
    kt = 1
    if (mean_tx > mean_bx or mean_tx < mean_bx):
        k_lt_lb = 1.0 * (mean_by - mean_ty) / (mean_bx - mean_tx)
    if (k_lt_lb == 1111):
        kt = 0
    ta = math.fabs(math.atan(k_lt_lb))
    alpha = 180 * ta / math.pi

    if (kt == 0 and lt_y < lb_y - 3):
        alpha = 0
    if (kt == 0 and lt_y > lb_y + 3):
        alpha = 180
    if (k_lt_lb == 0 and lt_x > lb_x):
        alpha = 90
    if (k_lt_lb == 0 and lt_x < lb_x):
        alpha = -90
    if (k_lt_lb > 0 and rt_y > lb_y):
        ta = math.fabs(math.atan(k_lt_lb))
        alpha = 180 * ta / math.pi + 90
    if (k_lt_lb > 0 and rt_y < lb_y):
        ta = math.fabs(math.atan(k_lt_lb))
        alpha = 180 * ta / math.pi - 90
    if (k_lt_lb < 0 and lt_y < rb_y):
        ta = math.fabs(math.atan(k_lt_lb))
        alpha = 90 - 180 * ta / math.pi
    if (k_lt_lb < 0 and lt_y > rb_y):
        ta = math.fabs(math.atan(k_lt_lb))
        alpha = -(180 * ta / math.pi + 90)

    return alpha

def detect_align(img_raw, detect_net):
    #读取图片
    # img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = np.float32(img_raw)

    #resize图片
    target_size = testconf.detect_size
    im_shape = img.shape
    im_size_max = np.max(im_shape[0:2])
    res_scal = 0.5
    if im_size_max < 300:
        res_scal = 0.75
    if im_size_max >= 300 and im_size_max < 600:
        res_scal = 0.375
    if im_size_max >= 600 and im_size_max < 800:
        res_scal = 0.25
    if im_size_max >= 800 and im_size_max < 1200:
        res_scal = 0.15
    if im_size_max >= 1200 and im_size_max < 2000:
        res_scal = 0.125
    if im_size_max >= 2000:
        res_scal = 0.1

    if testconf.origin_size:
        res_scal = 1

    if res_scal != 1:
        img = cv2.resize(img, None, None, fx = res_scal, fy = res_scal, interpolation=cv2.INTER_LINEAR)

    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    scale = scale.to(device)

    #减去均值转成numpy
    im_height, im_width, _ = img.shape
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)

    loc, conf, landms = detect_net(img)  # forward pass

    priorbox = prior_box.PriorBox(cfg_rfb, image_size=(im_height, im_width))
    prior_anchor = priorbox.forward()
    prior_anchor = prior_anchor.to(device)
    prior_data = prior_anchor.data
    boxes = decode(loc.data.squeeze(0), prior_data, [0.1, 0.2])
    boxes = boxes * scale / res_scal
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, [0.1, 0.2])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / res_scal
    landms = landms.cpu()
    landms = landms.detach().numpy()

    # ignore low scores
    inds = np.where(scores > testconf.confidence_thresh)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:6000]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, testconf.nms_thresh)
    dets = dets[keep, :]
    landms = landms[keep]

    return dets, landms

def get_right_box(rect, rects, points):
    wid = rect[2] - rect[0]
    hei = rect[3] - rect[1]
    rw0 = rects[0,2] - rects[0,0]
    rh0 = rects[0,3] - rects[0,1]
    minwh = (wid - rw0) * (wid - rw0) + (hei - rh0) * (hei - rh0)
    rightlands = points[0,:]
    rightrect = rects[0,:]
    for frect, fpoint in zip(rects, points):
        frw = frect[2] - frect[0]
        frh = frect[3] - frect[1]
        bxwh = (wid - frw) * (wid - frw) + (hei - frh) * (hei - frh)
        if bxwh < minwh:
            rightlands = fpoint
            rightrect = frect
    return rightrect, rightlands

def get_mask_box(rect, imgw, imgh):
    bx = rect[0]
    by = rect[1]
    bw = rect[2] - rect[0]
    bh = rect[3] - rect[1]

    randx1 = random.choice([0.2, 0.25, 0.3, 0.35])
    randy1 = random.choice([0.5, 0.54, 0.58, 0.6])
    randx2 = random.choice([1.4, 1.5, 1.6, 1.7])
    randy2 = random.choice([0.6, 0.62, 0.64, 0.65])
    nbx1 = bx - randx1 * bw
    nby1 = by + randy1 * bh
    nbx2 = nbx1 + randx2 * bw
    nby2 = nby1 + randy2 * bh

    # nbx1 = bx - 0.25 * bw
    # nby1 = by + 0.55 * bh
    # nbx2 = nbx1 + 1.5 * bw
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
    im = cv2.resize(img, (testconf.img_width, testconf.img_height), interpolation=cv2.INTER_CUBIC)
    im = im.astype(np.float32)
    im = (im - testconf.bgr_mean) / testconf.bgr_std
    im = im.transpose(2, 0, 1)
    im = torch.from_numpy(im)
    im = im.unsqueeze(0)
    im = im.to(device)
    return im

def test_img(imgp, savedir, rfb_net, mask_net, id):
    plist = imgp.split("/")
    img_name = plist[-1]
    img_mat = cv2.imread(imgp, cv2.IMREAD_COLOR)
    face_rect, key_points = detect_align(img_mat, rfb_net)
    if face_rect.shape[0] > 0:
        ii = 0
        for frect, fpoint in zip(face_rect, key_points):
            ii += 1
            pp = np.zeros(8, dtype=np.int32)
            pp[0] = fpoint[0]
            pp[1] = fpoint[1]
            pp[2] = fpoint[2]
            pp[3] = fpoint[3]
            pp[4] = fpoint[6]
            pp[5] = fpoint[7]
            pp[6] = fpoint[8]
            pp[7] = fpoint[9]
            face_ang = get_angle(pp)
            rotated = imutils.rotate(img_mat, face_ang)
            img_h, img_w, _ = rotated.shape
            show_rot = rotated.copy()
            rot_rect, rot_points = detect_align(rotated, rfb_net)
            if rot_rect.shape[0] > 0:
                true_rect, true_points = get_right_box(frect, rot_rect, rot_points)
                # x1 = random.randrange(int(true_rect[0]-10), int(true_rect[0]+10))
                # y1 = random.randrange(int(true_rect[1]-10), int(true_rect[1]+10))
                # x2 = random.randrange(int(true_rect[2]-10), int(true_rect[2]+10))
                # y2 = random.randrange(int(true_rect[3]-10), int(true_rect[3]+10))

                # x1 = random.randrange(60,100)
                # y1 = random.randrange(155, 195)
                # x2 = random.randrange(170, 210)
                # y2 = random.randrange(285, 325)
                # true_rect[0] = 64#64
                # true_rect[1] = 48#48
                # true_rect[2] = 256#256
                # true_rect[3] = 272#272
                mask_rect = get_mask_box(true_rect, img_w, img_h)

                mask_roi = rotated[mask_rect[1]:mask_rect[3], mask_rect[0]:mask_rect[2], :]
                # mask_roi = rotated[64:48, 256:272, :]
                mask_roi = cv2.resize(mask_roi, (testconf.img_width, testconf.img_height))
                mask_img = img_process(mask_roi)
                outputs = mask_net(mask_img)
                _, prediction = torch.max(outputs.data, 1)
                # prediction = mask_net(mask_img)

                cv2.rectangle(rotated, (mask_rect[0], mask_rect[1]), (mask_rect[2], mask_rect[3]), (255, 0, 0), 2)

                posx = int(mask_rect[0])
                posy = int(mask_rect[1] + 0.25 * (mask_rect[3] - mask_rect[1]))
                if prediction == 0:
                    cv2.putText(rotated, "Not Pass", (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 255), 5)
                if prediction == 1:
                    cv2.putText(rotated, "Pass", (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 255, 0), 5)

                cv2.rectangle(show_rot, (true_rect[0], true_rect[1]), (true_rect[2], true_rect[3]), (255, 0, 0), 2)
                cv2.circle(show_rot, (true_points[0], true_points[1]), 1, (0, 0, 255), 4)
                cv2.circle(show_rot, (true_points[2], true_points[3]), 1, (0, 255, 255), 4)
                cv2.circle(show_rot, (true_points[4], true_points[5]), 1, (255, 0, 255), 4)
                cv2.circle(show_rot, (true_points[6], true_points[7]), 1, (0, 255, 0), 4)
                cv2.circle(show_rot, (true_points[8], true_points[9]), 1, (255, 0, 0), 4)

                rot_detResult = np.hstack([show_rot, rotated])
                cv2.namedWindow('result', cv2.WINDOW_NORMAL)
                cv2.imshow('result', rot_detResult)
                cv2.waitKey(0)
                savep = savedir + "/{}".format(id) + "wss.jpg"
                if prediction == 0:
                    # cv2.imwrite(savep, mask_roi)
                    return 0
                else:
                    return 1

def test_dir(imgdir, savedir, rfb_net, mask_net):
    for root, dirs, files in os.walk(imgdir):
        img_num = 0
        for f in files:
            img_num += 1
            if img_num % 1 == 0:
                img_path = os.path.join(root, f)
                path_new = img_path.replace('\\', '/')
                ft = test_img(path_new, savedir, rfb_net, mask_net, 1)

def test_cropimg(imgdir, savedir, mask_net):
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    for root, dirs, files in os.walk(imgdir):
        img_num = 0
        for f in files:
            img_num += 1
            if img_num % 1 == 0:
                img_path = os.path.join(root, f)
                path_new = img_path.replace('\\', '/')
                img_mat = cv2.imread(path_new, cv2.IMREAD_COLOR)
                mask_roi = cv2.resize(img_mat, (testconf.img_width, testconf.img_height))
                mask_img = img_process(mask_roi)
                outputs = mask_net(mask_img)
                _, prediction = torch.max(outputs.data, 1)
                posx = int(20)
                posy = int(45)
                if prediction == 0:
                    cv2.putText(mask_roi, "Not", (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
                if prediction == 1:
                    cv2.putText(mask_roi, "Pass", (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)

                savep = savedir + "/" + f
                cv2.imwrite(savep, mask_roi)
                cv2.imshow('result', mask_roi)
                cv2.waitKey(0)

def name_to_path(imgdir, find_name):
    find_name = find_name[1:]
    subdirs = os.listdir(imgdir)
    num_ids = len(subdirs)
    for i in range(num_ids):
        subdir = os.path.join(imgdir, subdirs[i])
        files = os.listdir(subdir)
        for file in files:
            if find_name in file:
                paths = os.path.join(subdir, file)
                paths_new = paths.replace('\\', '/')
                subdir_new = subdir.replace('\\', '/')
                return paths_new, subdir_new

def get_error_imgs(imgdir, errordir, savedir):
    for root, dirs, files in os.walk(errordir):
        for f in files:
            find_img, dir_path = name_to_path(imgdir, f)
            img_mat = cv2.imread(find_img, cv2.IMREAD_COLOR)
            cv2.imwrite(os.path.join(savedir, f), img_mat)

def get_MouthMask_imgs(imgdir, maskdir, savedir):
    for root, dirs, files in os.walk(maskdir):
        for f in files:
            find_img, dir_path = name_to_path(imgdir, f)
            files2 = os.listdir(dir_path)
            for file2 in files2:
                path = os.path.join(dir_path, file2)
                path_new = path.replace('\\', '/')
                img_mat = cv2.imread(path_new, cv2.IMREAD_COLOR)
                cv2.imwrite(os.path.join(savedir, file2), img_mat)

def crop_mask(imgdir, savedir, detnet):
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            if file.endswith("png"):
                imgname = file.split("-create.png")[0]
                imgpath = imgdir + "/" + file
                imgdata = cv2.imread(imgpath, cv2.IMREAD_COLOR)
                show_rot = imgdata
                imh, imw, _ = imgdata.shape
                face_rect, key_points = detect_align(imgdata, detnet)
                ii = 0
                for frect, fpoint in zip(face_rect, key_points):
                    mask_rect = get_mask_box(frect, imw, imh)
                    mask_roi = imgdata[mask_rect[1]:mask_rect[3], mask_rect[0]:mask_rect[2], :]
                    mask_roi = cv2.resize(mask_roi, (testconf.img_width, testconf.img_height))
                    savename = str(ii) + imgname + ".jpg"
                    ii += 1
                    savepath = savedir + "/" + savename
                    cv2.imwrite(savepath, mask_roi)
                    cv2.rectangle(show_rot, (frect[0], frect[1]), (frect[2], frect[3]), (255, 0, 0), 2)
                    cv2.rectangle(show_rot, (mask_rect[0], mask_rect[1]), (mask_rect[2], mask_rect[3]), (0, 255, 0), 2)

                cv2.imshow('result', show_rot)
                cv2.waitKey(1)




# img_dir1 = "D:/data/imgs/facePicture/mask/clean"
# errordir1 = "D:/data/imgs/facePicture/mask/result2"
# savedir1 = "D:/data/imgs/facePicture/mask/kaoqing/3"
# # get_error_imgs(img_dir1, errordir1, savedir1)
# get_MouthMask_imgs(img_dir1, errordir1, savedir1)

rfb_net = RFB(phase='test')
model_home = testconf.model_save
retina_model = testconf.rfb_model
model_det = os.path.join(model_home, retina_model)
rfb_weight = torch.load(model_det)
rfb_net.load_state_dict(rfb_weight)
rfb_net.eval()
print('Finished loading rfb model!')
rfb_net = rfb_net.to(device)

mask_net = maskNet(n_class=testconf.num_classes)
slim_model = testconf.mask_model
model_classification = os.path.join(model_home, slim_model)
slim_weight = torch.load(model_classification, map_location={'cuda:1':'cuda:0'})
mask_net.load_state_dict(slim_weight)
mask_net.eval()
print('Finished loading slim model!')
mask_net = mask_net.to(device)

img_path = "imgs3/3.bmp"
save_dir = "D:/data/imgs/facePicture/mask/kaoqing/true"
payh2 = "D:/huoti_detect/imgdata/tfg_id"

# err = 0
# rangnum = 1
# for i in range(rangnum):
#     lab = test_img(img_path, save_dir, rfb_net, mask_net, i)
#     if(lab == 0):
#         err += 1
# err_rate = 1.0 * err / rangnum
# print("虚警率为: %.3f"%(err_rate))

img_dir = "D:/data/imgs/facePicture/mask/mask_matting"
dir_save = "D:/data/imgs/facePicture/mask/train/matting"
# test_dir(img_dir, dir_save, rfb_net, mask_net)
# test_cropimg(img_dir, dir_save, mask_net)
crop_mask(img_dir, dir_save, rfb_net)




































# imgPath = "imgs"
# img_name = "b6.jpg"
# img_mat = cv2.imread(os.path.join(imgPath, img_name), cv2.IMREAD_COLOR)
# face_rect, key_points = detect_align(img_mat, rfb_net)
# for frect, fpoint in zip(face_rect, key_points):
#     pp = np.zeros(8, dtype=np.int32)
#     pp[0] = fpoint[0]
#     pp[1] = fpoint[1]
#     pp[2] = fpoint[2]
#     pp[3] = fpoint[3]
#     pp[4] = fpoint[6]
#     pp[5] = fpoint[7]
#     pp[6] = fpoint[8]
#     pp[7] = fpoint[9]
#     face_ang = get_angle(pp)
#     rotated = imutils.rotate(img_mat, face_ang)
#     img_h, img_w, _ = rotated.shape
#     show_rot = rotated.copy()
#     rot_rect, rot_points = detect_align(rotated, rfb_net)
#     true_rect, true_points = get_right_box(frect, rot_rect, rot_points)
#     mask_rect = get_mask_box(true_rect, img_w, img_h)
#
#     mask_roi = rotated[mask_rect[1]:mask_rect[3], mask_rect[0]:mask_rect[2], :]
#     mask_roi = cv2.resize(mask_roi, (testconf.img_width, testconf.img_height))
#     mask_img = img_process(mask_roi)
#     outputs = mask_net(mask_img)
#     _, prediction = torch.max(outputs.data, 1)
#
#     cv2.rectangle(rotated, (mask_rect[0], mask_rect[1]), (mask_rect[2], mask_rect[3]), (255, 0, 0), 2)
#
#     posx = int(mask_rect[0])
#     posy = int(mask_rect[1] + 0.25 * (mask_rect[3] - mask_rect[1]))
#     if prediction == 0:
#         cv2.putText(rotated, "Not Pass", (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 255), 5)
#     if prediction == 1:
#         cv2.putText(rotated, "Pass", (posx, posy), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 255, 0), 5)
#
#
#
#
#     cv2.rectangle(show_rot, (true_rect[0], true_rect[1]), (true_rect[2], true_rect[3]), (255, 0, 0), 2)
#     cv2.circle(show_rot, (true_points[0], true_points[1]), 1, (0, 0, 255), 4)
#     cv2.circle(show_rot, (true_points[2], true_points[3]), 1, (0, 255, 255), 4)
#     cv2.circle(show_rot, (true_points[4], true_points[5]), 1, (255, 0, 255), 4)
#     cv2.circle(show_rot, (true_points[6], true_points[7]), 1, (0, 255, 0), 4)
#     cv2.circle(show_rot, (true_points[8], true_points[9]), 1, (255, 0, 0), 4)
#
#     rot_detResult = np.hstack([show_rot, rotated])
#     cv2.namedWindow('result', cv2.WINDOW_NORMAL)
#     cv2.imshow('result', rot_detResult)
#     save_path = "result"
#     cv2.imwrite(os.path.join(save_path, img_name), rot_detResult)
#     cv2.waitKey(0)































































