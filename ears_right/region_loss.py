import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

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

def build_label(pred_boxes, target, anchors, num_anchors, nH, nW, noobject_scale, object_scale, sil_thresh, seen=15000):
    nB = target.size(0) #batch=4
    nA = num_anchors #5
    anchor_step = int(len(anchors) / num_anchors) #2
    conf_mask = torch.ones(nB, nA, nH, nW) * noobject_scale
    coord_mask = torch.zeros(nB, nA, nH, nW)
    cls_mask = torch.zeros(nB, nA, nH, nW)
    tx = torch.zeros(nB, nA, nH, nW)
    ty = torch.zeros(nB, nA, nH, nW)
    tw = torch.zeros(nB, nA, nH, nW)
    th = torch.zeros(nB, nA, nH, nW)
    tconf = torch.zeros(nB, nA, nH, nW)
    tcls = torch.zeros(nB, nA, nH, nW)

    lr_mask = torch.zeros(nB, nA, nH, nW)
    tconf_lr = torch.zeros(nB, nA, nH, nW)

    nAnchors = nA * nH * nW
    nPixels = nH * nW
    for b in range(nB):
        cur_pred_boxes = pred_boxes[b * nAnchors:(b + 1) * nAnchors].t()
        cur_ious = torch.zeros(nAnchors)
        for t in range(50):
            if target[b][t * 6 + 1] == 0:
                break
            gx = target[b][t * 6 + 1] * nW
            gy = target[b][t * 6 + 2] * nH
            gw = target[b][t * 6 + 3] * nW
            gh = target[b][t * 6 + 4] * nH
            cur_gt_boxes = torch.FloatTensor([gx, gy, gw, gh]).repeat(nAnchors, 1).t()
            cur_ious = torch.max(cur_ious, bbox_iou(cur_pred_boxes, cur_gt_boxes, x1y1x2y2=False))
        cur_ious = cur_ious.view(nA, nH, nW)
        conf_mask[b][cur_ious > sil_thresh] = 0
    if seen < 12800:
        tx.fill_(0.5)
        ty.fill_(0.5)
        tw.zero_()
        th.zero_()
        coord_mask.fill_(1)

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(50):
            if target[b][t * 6 + 1] == 0:
                break
            nGT = nGT + 1
            best_iou = 0.0
            best_n = -1
            min_dist = 10000
            gx = target[b][t * 6 + 1] * nW
            gy = target[b][t * 6 + 2] * nH
            gi = int(gx)
            gj = int(gy)
            gw = target[b][t * 6 + 3] * nW
            gh = target[b][t * 6 + 4] * nH
            gt_box = torch.FloatTensor([0, 0, gw, gh])
            for n in range(nA):
                aw = anchors[anchor_step * n]
                ah = anchors[anchor_step * n + 1]
                anchor_box = torch.FloatTensor([0, 0, aw, ah])
                iou = bbox_iou(anchor_box, gt_box, x1y1x2y2=False)
                if iou > best_iou:
                    best_iou = iou
                    best_n = n

            gt_box = torch.FloatTensor([gx, gy, gw, gh])
            pred_box = pred_boxes[b * nAnchors + best_n * nPixels + gj * nW + gi]

            coord_mask[b][best_n][gj][gi] = 1
            cls_mask[b][best_n][gj][gi] = 1
            conf_mask[b][best_n][gj][gi] = object_scale
            tx[b][best_n][gj][gi] = target[b][t * 6 + 1] * nW - gi
            ty[b][best_n][gj][gi] = target[b][t * 6 + 2] * nH - gj
            tw[b][best_n][gj][gi] = math.log(gw / anchors[anchor_step * best_n])
            th[b][best_n][gj][gi] = math.log(gh / anchors[anchor_step * best_n + 1])
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)  # best_iou
            tconf[b][best_n][gj][gi] = iou

            lr_mask[b][best_n][gj][gi] = 1
            tconf_lr[b][best_n][gj][gi] = target[b][t * 6 + 5]

            tcls[b][best_n][gj][gi] = target[b][t * 6]
            if iou > 0.5:
                nCorrect = nCorrect + 1
    return nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls, lr_mask, tconf_lr

class loss_v2(nn.Module):
    def __init__(self, num_classes=2, anchors=[], num_anchors=5):
        super(loss_v2, self).__init__()
        self.num_classes = num_classes # 20,80
        self.anchors = anchors #40,15, 90,45, 120,55, 190,65, 220,88
        self.num_anchors = num_anchors #5
        self.anchor_step = int(len(anchors) / num_anchors) #2
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6

    def forward(self, output, target):
        # output : BxAs*(4+1+num_classes)*H*W
        nB = output.data.size(0) #batch=4
        nA = self.num_anchors #5
        nC = self.num_classes #2
        nH = output.data.size(2) #13
        nW = output.data.size(3) #13

        output = output.view(nB, nA, (5 + nC + 1), nH, nW)
        x = torch.sigmoid(output.index_select(2, torch.tensor([0]).cuda()).view(nB, nA, nH, nW))
        y = torch.sigmoid(output.index_select(2, torch.tensor([1]).cuda()).view(nB, nA, nH, nW))
        w = output.index_select(2, torch.tensor([2]).cuda()).view(nB, nA, nH, nW)
        h = output.index_select(2, torch.tensor([3]).cuda()).view(nB, nA, nH, nW)
        conf = torch.sigmoid(output.index_select(2, torch.tensor([4]).cuda()).view(nB, nA, nH, nW))
        cls = output.index_select(2, torch.linspace(5, 5 + nC - 1, nC).long().cuda())
        cls = cls.view(nB * nA, nC, nH * nW).transpose(1, 2).contiguous().view(nB * nA * nH * nW, nC)

        conf_lr = torch.sigmoid(output.index_select(2, torch.tensor([5 + nC]).cuda()).view(nB, nA, nH, nW))

        pred_boxes = torch.cuda.FloatTensor(4, nB * nA * nH * nW)
        # pred_boxes = torch.zeros(size = (4, nB * nA * nH * nW)).cuda()
        grid_x = torch.linspace(0, nW - 1, nW).repeat(nH, 1).repeat(nB * nA, 1, 1).view(nB * nA * nH * nW).cuda()
        grid_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA, 1, 1).view(nB * nA * nH * nW).cuda()
        anchor_w = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([0])).cuda()
        anchor_h = torch.Tensor(self.anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([1])).cuda()
        anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH * nW).view(nB * nA * nH * nW)
        anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH * nW).view(nB * nA * nH * nW)
        xv = x.view(nB * nA * nH * nW)
        yv = y.view(nB * nA * nH * nW)
        wv = w.view(nB * nA * nH * nW)
        hv = h.view(nB * nA * nH * nW)
        pred_boxes[0] = xv.data + grid_x
        pred_boxes[1] = yv.data + grid_y
        pred_boxes[2] = torch.exp(wv.data) * anchor_w
        pred_boxes[3] = torch.exp(hv.data) * anchor_h
        pred_boxes = convert2cpu(pred_boxes.transpose(0, 1).contiguous().view(-1, 4))

        nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls , lr_mask, tconf_lr\
            = build_label(pred_boxes, target.data, self.anchors, nA, nH, nW, self.noobject_scale, self.object_scale, self.thresh)

        cls_mask = (cls_mask == 1)
        nProposals = int((conf > 0.25).sum())

        tx = tx.cuda()
        ty = ty.cuda()
        tw = tw.cuda()
        th = th.cuda()
        tconf = tconf.cuda()
        tcls = tcls[cls_mask].view(-1).long().cuda()

        tconf_lr = tconf_lr.cuda()

        coord_mask = coord_mask.cuda()
        conf_mask = conf_mask.cuda().sqrt()
        cls_mask = cls_mask.view(-1, 1).repeat(1, nC).cuda()
        cls = cls[cls_mask].view(-1, nC)

        loss_x = self.coord_scale * nn.MSELoss(reduction='sum')(x * coord_mask, tx * coord_mask) / 2.0
        loss_y = self.coord_scale * nn.MSELoss(reduction='sum')(y * coord_mask, ty * coord_mask) / 2.0
        loss_w = self.coord_scale * nn.MSELoss(reduction='sum')(w * coord_mask, tw * coord_mask) / 2.0
        loss_h = self.coord_scale * nn.MSELoss(reduction='sum')(h * coord_mask, th * coord_mask) / 2.0
        loss_conf = nn.MSELoss(reduction='sum')(conf * conf_mask, tconf * conf_mask) / 2.0
        loss_cls = self.class_scale * nn.CrossEntropyLoss(reduction='sum')(cls, tcls)

        lr_mask = lr_mask.cuda()
        loss_conf_lr = nn.MSELoss(reduction='sum')(conf_lr * lr_mask, tconf_lr * lr_mask) / 2.0

        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls + 0.5 * loss_conf_lr

        # print('nGT %d, recall %d, proposals %d, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % (
        # nGT, nCorrect, nProposals, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(),
        # loss_conf.item(), loss_cls.item(), loss.item()))
        return loss / nB



