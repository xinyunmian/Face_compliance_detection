import shutil
from math import sqrt, tan
import cv2 as cv
import dlib
import numpy as np
from cv2.cv2 import cvtColor
import glob

src_path = "F://work//Exe//9-8//Left//Left//out_src2//"
seg_path = 'F://work//Exe//9-8//Left//Left//out2//'
save_dirpath = ""
file_path_list = glob.glob(src_path + '*.*g')

for image_path in file_path_list:
    # image_path = 'F://work//Exe//9-8//image_overdk//image_overdk//out_src//1416562_2.png'
    # 1、读取图片
    image_name = image_path.split('\\')[-1].split('.')[0]
    # print('图片%s已完成' % image_name)
    image_seg_path = seg_path + image_name + '-seg.png'

    img_ori = cv.imread(image_path)
    img_seg = cv.imread(image_seg_path)

    # 2、获取分割图中的面部区域
    face_mask = (img_seg == 1).astype(np.float32)
    # 求面部区域的和，因为值全为1，所以只需求值为1的长度
    rows_face, cols_face = np.nonzero(face_mask[:, :, 1])
    sum_face = len(rows_face)
    # 3、将face_mask与原始图片点乘
    img_src = img_ori.astype(np.float32)
    img_fin = np.multiply(img_src, face_mask) / 255.

    # 4、将BGR转换成LAB
    img_lab = cvtColor(img_fin, cv.COLOR_RGB2Lab)
    # cv.imwrite('./test_img.png', img_lab.astype(np.uint8))

    # 5.求L通道的均值，判断亮度
    l_channel, a_channel, b_channel = cv.split(img_lab)
    # 求出L通道的非零值的位置，然后对这些位置进行求和，随后再计算均值
    rows_l, cols_l = np.nonzero(l_channel)
    sum_l = sum(l_channel[rows_l, cols_l])
    mean_l = (sum_l / sum_face)

    # 6、判断阴阳脸
    dest = []
    for e in rows_l:
        if e not in dest:
            dest.append(e)

        # 找出中心点


    def center(img):
        detector = dlib.get_frontal_face_detector()

        # 使用官方提供的模型构建特征提取器
        predictor = dlib.shape_predictor(
            'F:\work\Exe\8_31\shape_predictor_68_face_landmarks\shape_predictor_68_face_landmarks.dat')

        # 与人脸检测程序相同,使用detector进行人脸检测 dets为返回的结果
        dets = detector(img, 1)
        # 使用enumerate 函数遍历序列中的元素以及它们的下标
        # 下标k即为人脸序号
        # left：人脸左边距离图片左边界的距离 ；right：人脸右边距离图片左边界的距离
        # top：人脸上边距离图片上边界的距离 ；bottom：人脸下边距离图片上边界的距离
        for k, d in enumerate(dets):
            # print("dets{}".format(d))
            # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            #     k, d.left(), d.top(), d.right(), d.bottom()))

            # 使用predictor进行人脸关键点识别 shape为返回的结果
            shape = predictor(img, d)
            # 获取第一个和第二个点的坐标（相对于图片而不是框出来的人脸）
            # print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))

            # 绘制特征点
            for index, pt in enumerate(shape.parts()):
                # print('Part {}: {}'.format(index, pt))
                pt_pos = (pt.x, pt.y)
                if index == 28:
                    return pt_pos


    x = center(img_ori)
    # 左半边脸
    if x is not None:
        left_l_channel = l_channel[dest, min(cols_l):x[1]]
        rows_left, cols_left = np.nonzero(left_l_channel)
        mean_1eft = sum(sum(left_l_channel)) / len(rows_left)

        # 右半边脸
        right_l_channel = l_channel[dest, x[1]:max(cols_l)]
        rows_right, cols_right = np.nonzero(right_l_channel)
        mean_right = sum(sum(right_l_channel)) / len(rows_right)
        left_right = mean_1eft - mean_right
        # print(sum(sum(left_l_channel)),sum(sum(right_l_channel)))

        # 7、判断是否偏色
        da = (sum(sum(a_channel))) / (a_channel.shape[0] * a_channel.shape[1])
        db = (sum(sum(b_channel))) / (b_channel.shape[0] * b_channel.shape[1])
        D1 = sqrt(da * da + db * db)
        theta1 = tan(db / da)

        ma = (sum(sum(abs(a_channel - da)))) / (a_channel.shape[0] * a_channel.shape[1])
        mb = (sum(sum(abs(b_channel - db)))) / (b_channel.shape[0] * b_channel.shape[1])
        D2 = sqrt(ma * ma + mb * mb)
        theta2 = tan(mb / ma)
        # 模比值(<1)
        k1 = D1 / D2
        # 角度差值([-Pi,Pi/2])
        k2 = theta1 / theta2

        dark = 5 #过暗
        bright = 5 #过亮
        yin_yang = 5 #阴阳
        # 进行分类
        if 60 <= mean_l <= 66:
            dark = 0 #亮度正常0
            bright = 0 #亮度正常0
        elif 50.0 <= mean_l < 60.0:
            dark = 1 #过暗30
            bright = 0
        elif 40.0 < mean_l < 50:
            dark = 2 #过暗70
            bright = 0
        elif mean_l < 40:
            dark = 3 #过暗100
            bright = 0
        elif 66 < mean_l <= 70.0:
            bright = 1  # 过亮30
            dark = 0
        elif 70.0 < mean_l <= 80.0:
            bright = 2  # 过亮70
            dark = 0
        else:
            bright = 3  # 过亮100
            dark = 0
        if 0 <= abs(left_right) <= 5:
            yin_yang = 0 #阴阳脸正常0
        elif 5 < abs(left_right) <= 10:
            yin_yang = 1 #阴阳脸30
        elif 10 < abs(left_right) <= 15:
            yin_yang = 2 #阴阳脸70
        else:
            yin_yang = 3  # 阴阳脸100

        label = "_" + str(dark) + "_" + str(bright) + "_" + str(yin_yang)
        target_path = save_dirpath + "/" + image_name + label + '.png'
        shutil.copyfile(image_path, target_path)

            # print("image_name:%s,亮度值：%f,左右脸差值：%f,模比值：%f,角度差值：%f" % (image_name, mean_l, left_right, k1, k2))
    else:
        print("问题图片image_name:%s" % (image_name))





