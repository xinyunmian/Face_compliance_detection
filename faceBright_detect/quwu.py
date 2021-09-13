from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk
import skimage.filters.rank as sfr
import cv2

def min_box(image, kernel_size=15):
    min_image = sfr.minimum(image, disk(kernel_size))
    return min_image

def zmMinFilterGray(src, r=7):
    '''''最小值滤波，r是滤波器半径'''
    return cv2.erode(src,np.ones((2*r-1,2*r-1)))

def calculate_dark(image):
    """
    get dark channel
    Args:
        image: numpy type image
    Returns:
        dark channel
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("input image is not numpy type")
    dark = np.minimum(np.minimum(image[:, :, 0], image[:, :, 1]), image[:, :, 2]).astype(np.float32)
    # if kernel_size too small, will get too
    # bright dark channel, and also make A too bigger
    # min_box will lead:  dark = dark * 255
    dark = zmMinFilterGray(dark, r=7)
    # dark = min_box(dark, kernel_size=15)

    return dark

imgpath1 = "D:/data/imgs/facePicture/face_bright/face_skin/youwu/16.jpg"
imgpath2 = "D:/data/imgs/facePicture/face_bright/face_skin/youwu/011208.png"
savepath = "D:/codes/project/ageTest/result/af93c266db7b11eaad2600163e0070b6.jpg"

# haze = np.array(Image.open(imgpath1))[:, :, 0:3]/255
# clear = np.array(Image.open(imgpath2))[:, :, 0:3]/255
# dark_haze = calculate_dark(haze)
# dark_clear = calculate_dark(clear)
# plt.figure()
# plt.subplot(2, 2, 1)
# plt.imshow(haze)
# plt.subplot(2, 2, 2)
# plt.imshow(dark_haze, cmap="gray")
# plt.subplot(2, 2, 3)
# plt.imshow(clear)
# plt.subplot(2, 2, 4)
# plt.imshow(dark_clear, cmap="gray")
# plt.show()




im_no = cv2.imread(imgpath1) / 255
im_wu = cv2.imread(imgpath2) / 255

dark_no = calculate_dark(im_no)
# no_gray = cv2.cvtColor(dark_no, cv2.COLOR_GRAY2BGR)
dark_wu = calculate_dark(im_wu)
# wu_gray = cv2.cvtColor(dark_wu, cv2.COLOR_GRAY2BGR)

cv2.namedWindow('no', cv2.WINDOW_NORMAL)
cv2.imshow('no', dark_no)
cv2.namedWindow('wu', cv2.WINDOW_NORMAL)
cv2.imshow('wu', dark_wu)
cv2.waitKey(0)














