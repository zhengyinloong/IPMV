# -*- coding:utf-8 -*-
# demo2-hough.py
# zhengyinloong
# 2023/04/27 17:15
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mycv import *

def edge_detecting(img):
    # Gray
    # 高斯降噪
    sigma = 10
    k_size = 3
    img = cv.GaussianBlur(img, (k_size, k_size), sigma)

    # canny检测
    edges = mycanny(img, flag=3)
    return edges


def hough_lines(img):

    lines = np.zeros((3,3))
    return lines
    pass


def line_detect(img):
    edges = edge_detecting(img)
    # 霍夫检测
    # lines = hough_lines(img)
    # # print(lines)
    # print("Line Num : ", len(lines))
    #
    # # 画出检测的线段
    # for line in lines:
    #     for x1, y1, x2, y2 in line:
    #         cv.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
    #     pass
    return img_with_lines

# Read image

IMG1 = "10.jpg"
img1 = cv.imread(IMG1, cv.IMREAD_GRAYSCALE)

# Hough
drho = 1
d = 1
A = np.zeros((200,200),dtype=np.uint)

img1 = line_detect(img1)
cv.imshow("Hough",img1)


# Wait until a key pressed
cv.waitKey(0)

# Destroy all the windows opened before
cv.destroyAllWindows()




