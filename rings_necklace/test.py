import torch
import os
from mobile_yolo2 import SlimNet
from train_config import traincfg
import cv2
import random
import shutil
import numpy as np


device = "cpu"

def img_process(img):
    """将输入图片转换成网络需要的tensor
            Args:
                img_path: 人脸图片路径
            Returns:
                tensor： img(batch, channel, width, height)
    """
    im = cv2.resize(img, (traincfg.netw, traincfg.neth), interpolation=cv2.INTER_LINEAR)
    im = im.astype(np.float32)
    # im = (im - traincfg.bgr_mean) / traincfg.bgr_std
    im = im / 255.0
    im = im.transpose(2, 0, 1)
    im = torch.from_numpy(im)
    im = im.unsqueeze(0)
    im = im.to(device)
    return im

def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)

def bbox_iou(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = torch.min(boxes1[0], boxes2[0])
        Mx = torch.max(boxes1[2], boxes2[2])
        my = torch.min(boxes1[1], boxes2[1])
        My = torch.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = torch.min(boxes1[0] - boxes1[2] / 2.0, boxes2[0] - boxes2[2] / 2.0)
        Mx = torch.max(boxes1[0] + boxes1[2] / 2.0, boxes2[0] + boxes2[2] / 2.0)
        my = torch.min(boxes1[1] - boxes1[3] / 2.0, boxes2[1] - boxes2[3] / 2.0)
        My = torch.max(boxes1[1] + boxes1[3] / 2.0, boxes2[1] + boxes2[3] / 2.0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea / uarea

def plot_boxes_cv2(img, boxes):
    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = (box[0] - box[2]/2.0) * width
        y1 = (box[1] - box[3]/2.0) * height
        x2 = (box[0] + box[2]/2.0) * width
        y2 = (box[1] + box[3]/2.0) * height
        cls_id = str(int(box[6]))
        img = cv2.rectangle(img, (x1.data, y1.data), (x2.data, y2.data), (0, 255, 255), 2)
        img = cv2.putText(img, cls_id, (x1.data, y1.data), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
    return img

def get_boxes_yolo(output, anchors, num_anchors, num_classes, conf_thresh, use_sigmoid):
    anchor_step = len(anchors) // num_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch_size = output.size(0)
    assert output.size(1) == (5 + num_classes) * num_anchors
    h = output.size(2)
    w = output.size(3)
    all_boxes = []
    output = output.view(batch_size*num_anchors, 5+num_classes, h*w).transpose(0, 1).contiguous().view(5+num_classes, batch_size*num_anchors*h*w)
    grid_x = torch.linspace(0, w-1, w).repeat(h, 1).repeat(batch_size*num_anchors, 1, 1).view(batch_size*num_anchors*h*w).type_as(output)
    grid_y = torch.linspace(0, h-1, h).repeat(w, 1).t().repeat(batch_size*num_anchors, 1, 1).view(batch_size*num_anchors*h*w).type_as(output)
    xs = torch.sigmoid(output[0]) + grid_x
    ys = torch.sigmoid(output[1]) + grid_y
    anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
    anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
    anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, h*w).view(batch_size*num_anchors*h*w).type_as(output)
    anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, h*w).view(batch_size*num_anchors*h*w).type_as(output)
    ws = torch.exp(output[2]) * anchor_w
    hs = torch.exp(output[3]) * anchor_h
    if use_sigmoid:
        cls_confs = torch.nn.Sigmoid()(torch.autograd.Variable(output[5: 5+num_classes].transpose(0, 1))).data
    else:
        cls_confs = torch.nn.Softmax(dim=1)(torch.autograd.Variable(output[5: 5+num_classes].transpose(0, 1))).data
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)
    sz_hw = h * w
    sz_hwa = sz_hw * num_anchors
    det_confs = torch.sigmoid(output[4])
    det_confs = convert2cpu(det_confs)

    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs = convert2cpu(xs)
    ys = convert2cpu(ys)
    ws = convert2cpu(ws)
    hs = convert2cpu(hs)
    for b in range(batch_size):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b*sz_hwa + i*sz_hw + cy*w + cx
                    det_conf = det_confs[ind]
                    if det_conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [bcx/w, bcy/h, bw/w, bh/h, det_conf, cls_max_conf, cls_max_id]
                        boxes.append(box)
        all_boxes.append(boxes)
    return all_boxes

def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes
    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = boxes[i][4]
    # descending
    _, sortIds = torch.sort(det_confs, descending=True)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i+1, len(boxes)):
                box_j = boxes[sortIds[j]]
                if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                    box_j[4] = 0
    return out_boxes

def test_one(img_mat, dnet):
    img = img_process(img_mat)  # 数据处理，转为网络输入的形式
    outdata = dnet(img)
    output = outdata.data
    boxes = get_boxes_yolo(output, conf_thresh=traincfg.conf_thresh, num_classes=traincfg.label_class, anchors=traincfg.anchors,
                    num_anchors=traincfg.nanchors, use_sigmoid=False)
    boxes = boxes[0]
    bboxes = nms(boxes, traincfg.nms_thresh)
    draw_img = plot_boxes_cv2(img_mat, bboxes)
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.imshow('result', draw_img)
    cv2.waitKey(0)

if __name__ == "__main__":
    net = SlimNet(cfg=traincfg)
    weight_path = "weights/rings_necklaces.pth"
    net_dict = torch.load(weight_path, map_location=lambda storage, loc: storage)
    net.load_state_dict(net_dict)
    net.eval()
    print('Finished loading model!')
    net = net.to(device)

    # from save_params import pytorch_to_dpcoreParams
    # saveparams = pytorch_to_dpcoreParams(net)
    # saveparams.forward("eartype_param_cfg.h", "eartype_param_src.h")

    img_path = "D:/data/imgs/facePicture/glasses/test/decoration_210818/FN/positive_6.jpg"
    img = cv2.imread(img_path)
    test_one(img, net)






