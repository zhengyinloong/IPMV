# -*- coding:utf-8 -*-
# OTSUtest.py
# zhengyinloong
# 2023/05/03 19:40

"""
OTSUtest
"""
import numpy as np

from mycv import *

# IMG = "./10.jpg"
IMG = "./kunkun.jpg"
# 读取图像
gray_img = cv.imread(IMG, cv.IMREAD_GRAYSCALE)
Otsu = np.zeros_like(gray_img)

thresholds = otsu_plus(gray_img,4)
print(thresholds)
cv.imshow('i',gray_img)
# 根据阈值分割
for i,t in enumerate(thresholds):
    if i == 0:
        Otsu[gray_img>=t]=255
        continue
    elif t >= np.min(thresholds):
        for j,line in enumerate(gray_img):
            for k,p in enumerate(line):
                if  thresholds[i]<=p<thresholds[i-1]:
                    Otsu[j,k]=t

cv.imshow('OTSU', Otsu)

cv.waitKey(0)
cv.destroyAllWindows()
# # OTSU算法获取二值化阈值
# ret, thresh = cv.threshold(gray_img, 0, 255, cv.THRESH_OTSU)
# # 显示原始图像和分割后的图像
# cv.imshow('original', gray_img)
# cv.imshow('OTSU1', thresh)
# cv.waitKey(0)
# cv.destroyAllWindows()


