from __future__ import print_function
import os
import cv2
import numpy as np
from yolo_config import *
from save_params import save_feature_channel, pytorch_to_dpcoreParams
from test import img_process, get_boxes_yolo, plot_boxes_cv2, nms
from train_config import traincfg

from get_onnx import load_model
from onnxsim import simplify
import onnxruntime  # to inference ONNX models, we use the ONNX Runtime
import onnx

device = torch.device("cpu")

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        N = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        x = F.avg_pool2d(x, (H, W))
        x = x.view(N, C)
        return x

class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert (x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert (H % stride == 0)
        assert (W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, H // hs, hs, W // ws, ws).transpose(3, 4).contiguous()
        x = x.view(B, C, (H // hs) * (W // ws), hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, H // hs, W // ws).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, H // hs, W // ws)
        return x

class maxpool1(nn.Module):
	def __init__(self):
		super(maxpool1, self).__init__()
	def forward(self, x):
		x_pad = F.pad(x, (0, 1, 0, 1), mode='replicate')
		x = F.max_pool2d(x_pad, 2, stride=1)
		return x

class Upsample_expand(nn.Module):
    def __init__(self, stride=2):
        super(Upsample_expand, self).__init__()
        self.stride = stride
    def forward(self, x):
        assert (x.data.dim() == 4)
        B = x.size(0)
        C = x.size(1)
        H = x.size(2)
        W = x.size(3)
        x = x.view(B, C, H, 1, W, 1)
        x = x.expand(B, C, H, self.stride, W, self.stride).contiguous()
        x = x.view(B, C, H * self.stride, W * self.stride)
        return x

class Upsample_interpolate(nn.Module):
    def __init__(self, stride):
        super(Upsample_interpolate, self).__init__()
        self.stride = stride
    def forward(self, x):
        assert (x.data.dim() == 4)
        B = x.size(0)
        C = x.size(1)
        H = x.size(2)
        W = x.size(3)
        out = F.interpolate(x, size=(H * self.stride, W * self.stride), mode='nearest')
        return out

# for route and shortcut
class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()
    def forward(self, x):
        return x

class darknetCfg_to_pytorchModel(nn.Module):
    def __init__(self, cfgfile, count=5, mode="train"):
        super(darknetCfg_to_pytorchModel, self).__init__()
        self.header_len = count
        self.header = torch.IntTensor([0, ] * self.header_len)
        self.seen = self.header[3]
        self.mode = mode
        self.det_strides = []
        self.net_blocks = parse_cfg(cfgfile)
        self.models = self.create_net(self.net_blocks)  # merge conv, bn,leaky

    def create_net(self, blocks):
        models = nn.ModuleList()
        prev_filters = 3
        out_filters = []
        prev_stride = 1
        out_strides = []
        conv_id = 0

        for block in blocks:
            if block['type'] == 'net':
                init_width = int(block['width'])
                init_height = int(block['height'])
                prev_filters = int(block['channels'])
                continue
            elif block['type'] == 'convolutional':
                conv_id = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = (kernel_size - 1) // 2 if is_pad else 0
                group = 1
                if "groups" in block:
                    group = int(block['groups'])
                activation = block['activation']
                model = nn.Sequential()
                #conv
                if batch_normalize:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, groups=group, bias=False))
                    model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                else:
                    model.add_module('conv{0}'.format(conv_id), nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, groups=group))
                # activate function
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(conv_id), nn.ReLU(inplace=True))
                elif activation == 'mish':
                    model.add_module('mish{0}'.format(conv_id), Mish())
                else:
                    print("convalution no activate {}".format(activation))

                prev_filters = filters
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)

            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                if stride > 1:
                    model = nn.MaxPool2d(kernel_size=pool_size, stride=stride)
                else:
                    model = maxpool1()
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)

            elif block['type'] == 'avgpool':
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                models.append(model)

            elif block['type'] == 'softmax':
                model = nn.Softmax()
                out_strides.append(prev_stride)
                out_filters.append(prev_filters)
                models.append(model)

            elif block['type'] == 'upsample':
                stride = int(block['stride'])
                out_filters.append(prev_filters)
                prev_stride = prev_stride // stride
                out_strides.append(prev_stride)
                # models.append(Upsample_expand(stride))
                models.append(Upsample_interpolate(stride))

            elif block['type'] == 'shortcut':
                ind = len(models)
                prev_filters = out_filters[ind - 1]
                out_filters.append(prev_filters)
                prev_stride = out_strides[ind - 1]
                out_strides.append(prev_stride)
                models.append(EmptyModule())

            elif block['type'] == 'connected':
                filters = int(block['output'])
                stride = out_strides[-1]
                in_filters = prev_filters * (init_height//stride) * (init_width//stride)
                if block['activation'] == 'linear':
                    model = nn.Linear(in_filters, filters)
                elif block['activation'] == 'leaky':
                    model = nn.Sequential(
                        nn.Linear(in_filters, filters),
                        nn.LeakyReLU(0.1, inplace=True))
                elif block['activation'] == 'relu':
                    model = nn.Sequential(
                        nn.Linear(in_filters, filters),
                        nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(model)

            elif block['type'] == 'dropout':
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                prob = float(block['probability'])
                model = nn.Dropout(p=prob)
                models.append(model)
        return models

    def print_network(self):
        print_cfg(self.net_blocks)

    # load weights
    def load_weights(self, weightfile):
        with open(weightfile, 'rb') as fp:
            # before yolo3, weights get from https://github.com/pjreddie/darknet count = 4.
            header = np.fromfile(fp, count=self.header_len, dtype=np.int32)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]
            buf = np.fromfile(fp, dtype=np.float32)
        start = 0
        ind = -2
        for block in self.net_blocks:
            if start >= buf.size:
                break
            ind = ind + 1
            if block['type'] in ['net', 'maxpool', 'reorg', 'upsample', 'route', 'shortcut',
                                       'region', 'yolo', 'avgpool', 'softmax', 'cost', 'detection', 'dropout']:
                continue
            elif block['type'] in ['convolutional', 'local']:
                model = self.models[ind]
                try:
                    batch_normalize = int(block['batch_normalize'])
                except:
                    batch_normalize = False
                if batch_normalize:
                    start = load_conv_bn(buf, start, model[0], model[1])
                else:
                    start = load_conv(buf, start, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    start = load_fc(buf, start, model[0])
                else:
                    start = load_fc(buf, start, model)
            else:
                print('[Error]:unkown layer_type <%s>...' % (block['layer_type']))
                sys.exit(0)

    # save weights
    def save_weights(self, outfile):
        fp = open(outfile, 'wb')
        self.header[3] = self.seen
        header = self.header
        header.numpy().tofile(fp)
        ind = -1
        for blockId in range(1, len(self.net_blocks)):
            ind = ind + 1
            block = self.net_blocks[blockId]
            if block['type'] in ['convolutional', 'local']:
                model = self.models[ind]
                try:
                    batch_normalize = int(block['batch_normalize'])
                except:
                    batch_normalize = False
                if batch_normalize:
                    save_conv_bn(fp, model[0], model[1])
                else:
                    save_conv(fp, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    save_fc(fp, model[0])
                else:
                    save_fc(fp, model)
            elif block['type'] in ['net', 'maxpool', 'reorg', 'upsample', 'route', 'shortcut',
                                         'region', 'yolo', 'avgpool', 'softmax', 'cost', 'detection', 'dropout']:
                continue
            else:
                print('[Error]:unkown layer_type <%s>...' % (block['layer_type']))
                sys.exit(0)
        fp.close()

    def forward(self, x):
        ind = -2
        loss = 0
        res = []
        outputs = dict()
        for block in self.net_blocks:
            ind += 1
            if block['type'] == 'net':
                continue
            elif block['type'] in ['convolutional', 'maxpool', 'reorg', 'upsample', 'avgpool', 'softmax', 'local', 'dropout']:
                x = self.models[ind](x)
                if ind >= 10000:
                    b, c, h, w = x.shape
                    save_feature_channel("txt/conv1p.txt", x, b, c, h, w)
                outputs[ind] = x
            elif block['type'] == 'connected':
                x = x.view(x.size(0), -1)
                x = self.models[ind](x)
                outputs[ind] = x
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                route_groups = int(block['groups']) if "groups" in block else 1
                layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
                if len(layers) == 1:
                    if route_groups==1:
                        x = outputs[layers[0]]
                    elif route_groups==2:
                        x = outputs[layers[0]]
                        _, xc, _, _ = x.shape
                        x = x[:, xc // 2:, ...]
                    outputs[ind] = x
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x = torch.cat((x1, x2), 1)
                    outputs[ind] = x
            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                activation = block['activation']
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind - 1]
                x = x1 + x2
                if activation == 'leaky':
                    x = F.leaky_relu(x, 0.1, inplace=True)
                elif activation == 'relu':
                    x = F.relu(x, inplace=True)
                elif activation == 'mish':
                    x = Mish()(x)
                outputs[ind] = x
            # yoloV1, yoloV2, yoloV3
            elif block['type'] in ['detection', 'region', 'yolo']:
                res.append(x)
            else:
                print('[Error]:unkown layer_type <%s>...' % (block['layer_type']))
                sys.exit(0)
        return x if len(res) < 2 else res

def test_one(img_mat, dnet, dir=False):
    h, w, c = img_mat.shape
    img = img_process(img_mat)  # 数据处理，转为网络输入的形式
    outdata = dnet(img)
    output = outdata.data
    boxes = get_boxes_yolo(output, conf_thresh=traincfg.conf_thresh, num_classes=traincfg.label_class, anchors=traincfg.anchors,
                    num_anchors=traincfg.nanchors, use_sigmoid=False)
    boxes = boxes[0]
    bboxes = nms(boxes, traincfg.nms_thresh)
    if len(bboxes) > 0:
        draw_img = plot_boxes_cv2(img_mat, bboxes)
    else:
        draw_img = cv2.putText(img_mat, "normal", (int(0.3 * w), int(0.3 * h)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
    if dir:
        return draw_img
    else:
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow('result', draw_img)
        cv2.waitKey(0)

def testdir(tdir, sdir, dnet):
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    for root, dirs, files in os.walk(tdir):
        for file in files:
            root = root.replace("\\", "/")
            rootsplit = root.split("/")
            zidir = rootsplit[-1]
            imgpath = root + "/" + file
            savepath = sdir + "/" + file
            img = cv2.imread(imgpath)
            simg = test_one(img, dnet, dir=True)
            cv2.imwrite(savepath, simg)
            cv2.imshow('result', simg)
            cv2.waitKey(1)

def transform_onnx(net, size, weightp, onnxp, sim_onnxp):
    net = load_model(net, weightp, True)
    net.eval()
    print('Finished loading model!')
    net = net.to(device)

    print("==> Exporting model to ONNX format at '{}'".format(onnxp))
    input_names = ["img"]
    output_names = ["out"]
    inputs = torch.randn(1, 3, size, size).to(device)
    torch_out = torch.onnx._export(net, inputs, onnxp, export_params=True, verbose=False, input_names=input_names, output_names=output_names)

    onnx_model = onnx.load(onnxp)  # load onnx model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, sim_onnxp)
    print('finished exporting onnx')

if __name__ == "__main__":
    weightfile = "weights/rings_necklaces_200000.weights"
    cfgfile = "weights/rings_necklaces.cfg"

    pthp = "weights/rings_necklaces.pth"
    output_onnx = 'D:/codes/pytorch_projects/pytorch2ncnn/ring_lace.onnx'
    sim_onnx = 'D:/codes/pytorch_projects/pytorch2ncnn/ring_lace_simplify.onnx'

    model = darknetCfg_to_pytorchModel(cfgfile, mode="test")

    model.load_weights(weightfile)
    model.eval()

    # from save_params import pytorch_to_dpcoreParams
    # saveparams = pytorch_to_dpcoreParams(model)
    # saveparams.forward("jewelry_param_cfg.h", "jewelry_param_src.h")
    # torch.save(model.state_dict(), pthp)
    # transform_onnx(model, 320, pthp, output_onnx, sim_onnx)

    img_path = "D:/data/imgs/facePicture/glasses/test/decoration_210818/FN/positive_6.jpg"
    img = cv2.imread(img_path)
    # test_one(img, model)

    imgdir = "D:/data/imgs/facePicture/glasses/test/decoration_210818/FP"
    savedir = "D:/data/imgs/facePicture/glasses/test/decoration_210818/result/FP"
    testdir(imgdir, savedir, model)
