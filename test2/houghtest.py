# -*- coding:utf-8 -*-
# houghtest.py
# zhengyinloong
# 2023/04/28 07:29
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

IMG1 = "./10.jpg"
img1 = cv.imread(IMG1,cv.IMREAD_GRAYSCALE)

edge1 = edge_detecting(img1)

# Hough

A = 1


# Wait until a key pressed
cv.waitKey(0)

# Destroy all the windows opened before
cv.destroyAllWindows()

