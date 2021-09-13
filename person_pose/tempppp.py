import cv2
import os
import numpy as np
import random
from sklearn.decomposition import PCA

imgpath = "D:/data/imgs/facePicture/pose_person/shouder/4/6"
txtpath = 'D:/data/imgs/facePicture/pose_person/shouder/txt/'
savepath1 = "D:/data/imgs/facePicture/pose_person/shouder/1/"
savepath2 = "D:/data/imgs/facePicture/pose_person/shouder/2/"
savepath3 = "D:/data/imgs/facePicture/pose_person/shouder/3/"
im_names = os.listdir(imgpath)
for img_name in im_names:
    name = os.path.splitext(img_name)[0]
    raw_img = cv2.imread(imgpath + '/' + img_name)
    H, W = raw_img.shape[:2]
    loc = []
    with open(txtpath + name + '.txt', 'r') as f:
        i = 0
        for line in f.readlines():
            if i > 0:
                [x, y] = line.rstrip('\n').split(',')
                loc.append([int(x), int(y)])
            i += 1

    # angle = 8
    angle = random.choice([5, 6, 7, 8, 9, 10, 11, 12, 13])
    radian = angle / 180 * np.pi

    jaw = loc[8]

    # face_ctrl = np.array(loc, np.float32)
    face_ctrl = []
    x_ax = np.linspace(0, W - 1, 10)
    y_ax = np.linspace(0, jaw[1], 10)
    coors = np.meshgrid(x_ax, y_ax)
    x_coor, y_coor = coors[0], coors[1]
    x_coor = np.array(x_coor, np.float32)
    y_coor = np.array(y_coor, np.float32)

    for i in range(1, len(x_ax)):
        for j in range(1, len(x_coor[0]) - 1):
            face_ctrl.append([x_coor[i, j], y_coor[i, j]])
    face_ctrl = np.array(face_ctrl, np.float32)

    x_ax = np.linspace(0, W - 1, 15)
    y_ax = np.linspace(jaw[1] + 10, H - 1, 6)
    coors = np.meshgrid(x_ax, y_ax)
    x_coor, y_coor = coors[0], coors[1]
    x_coor = np.array(x_coor, np.float32)
    y_coor = np.array(y_coor, np.float32)

    body_ctrl = []
    for i in range(1, len(x_coor) - 1):
        for j in range(2, len(x_coor[0]) - 2):
            body_ctrl.append([x_coor[i, j], y_coor[i, j]])
    body_ctrl = np.array(body_ctrl, np.float32)

    y_center = (jaw[1] + H) // 2
    center = [W // 2, y_center]
    center = np.array(center, np.float32)
    body_temp, body_rot = body_ctrl.copy(), body_ctrl.copy()

    for i in range(len(body_rot)):
        body_temp[i, 0] -= center[0]
        body_temp[i, 1] -= center[1]

        body_rot[i, 0] = np.cos(radian) * body_temp[i, 0] - np.sin(radian) * body_temp[i, 1] + center[0]
        body_rot[i, 1] = np.sin(radian) * body_temp[i, 0] + np.cos(radian) * body_temp[i, 1] + center[1]

    border = []
    border_x = np.linspace(0, W - 1, 10)
    for i in border_x:
        border.append([i, 0])
        border.append([i, H - 1])

    border_y = np.linspace(0, H - 1, 10)
    for i in border_y[1:-1]:
        border.append([0, i])
        border.append([W - 1, i])
    border = np.array(border, np.float32)

    s_shape = np.concatenate((face_ctrl, border, body_ctrl), axis=0)
    t_shape = np.concatenate((face_ctrl, border, body_rot), axis=0)
    # s_shape = np.concatenate((border, body_ctrl), axis=0)
    # t_shape = np.concatenate((border, body_rot), axis=0)

    dots = t_shape.shape[0]

    s_shape = s_shape.reshape(1, -1, 2)
    t_shape = t_shape.reshape(1, -1, 2)

    matches = []
    for i in range(1, dots + 1):
        matches.append(cv2.DMatch(i, i, 0))

    tps = cv2.createThinPlateSplineShapeTransformer()
    tps.estimateTransformation(t_shape, s_shape, matches)

    out_img = tps.warpImage(raw_img)
    if angle > 4 and angle <= 7:
        cv2.imwrite(savepath1 + img_name, out_img)
    if angle > 7 and angle <= 10:
        cv2.imwrite(savepath2 + img_name, out_img)
    if angle > 10 and angle <= 13:
        cv2.imwrite(savepath3 + img_name, out_img)
    print(name)
