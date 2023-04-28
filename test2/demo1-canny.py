# -*- coding:utf-8 -*-
# demo1-canny.py
# zhengyinloong
# 2023/04/21 11:06

import cv2 as cv
from mycv import mycanny

IMG1 = './10.jpg'
IMG2 = './11.jpg'
IMG3 = './12.jpg'

img1 = cv.imread(IMG1, cv.IMREAD_GRAYSCALE)
img2 = cv.imread(IMG2, cv.IMREAD_GRAYSCALE)
img3 = cv.imread(IMG3, cv.IMREAD_GRAYSCALE)
# cv.imshow('Gray',img)
# 高斯降噪
sigma = 10
k_size = 3
img1 = cv.GaussianBlur(img1, (k_size, k_size), sigma)
img2 = cv.GaussianBlur(img2, (k_size, k_size), sigma)
img3 = cv.GaussianBlur(img3, (k_size, k_size), sigma)

# 二值化边缘
LowTh = 55
ratio = 4
cannyRet1 = mycanny(img1, LowTh, ratio, flag=3)
cannyRet2 = mycanny(img2, LowTh, ratio, flag=3)
cannyRet3 = mycanny(img3, LowTh, ratio, flag=3)
# 显示
cv.imshow('img1', img1)
cv.imshow('img2', img2)
cv.imshow('img3', img3)
cv.imshow('cannyRet1', cannyRet1)
cv.imshow('cannyRet2', cannyRet2)
cv.imshow('cannyRet3', cannyRet3)
cv.resizeWindow('img1', 350, 350)
cv.resizeWindow('img2', 350, 350)
cv.resizeWindow('img3', 350, 350)
cv.resizeWindow('cannyRet1', 350, 350)
cv.resizeWindow('cannyRet2', 350, 350)
cv.resizeWindow('cannyRet3', 350, 350)
# 保存
cv.imwrite('canny1.jpg', cannyRet1, [cv.IMWRITE_JPEG_QUALITY, 100])
cv.imwrite('canny2.jpg', cannyRet2, [cv.IMWRITE_JPEG_QUALITY, 100])
cv.imwrite('canny3.jpg', cannyRet3, [cv.IMWRITE_JPEG_QUALITY, 100])

# cv2官方Canny边缘检测函数
img1 = cv.imread(IMG1, cv.IMREAD_GRAYSCALE)
cannyRet = cv.Canny(img1,LowTh,LowTh*ratio)
cv.imshow('cannyRet', cannyRet)
cv.resizeWindow('cannyRet', 350, 350)

cv.waitKey(0)
cv.destroyAllWindows()
