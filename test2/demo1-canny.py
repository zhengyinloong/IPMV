# -*- coding:utf-8 -*-
# demo1-canny.py
# zhengyinloong
# 2023/04/21 11:06
"""
Canny边缘检测
"""
from mycv import *
if __name__=="__main__":

    IMG1 = "./10.jpg"
    IMG2 = "./11.jpg"
    IMG3 = "./12.jpg"
    IMGS = [IMG1,IMG2,IMG3]
    for i,IMG in enumerate(IMGS):
        img1 = cv.imread(IMG, cv.IMREAD_GRAYSCALE)
        # 高斯降噪
        sigma = 10
        k_size = 3
        img1 = cv.GaussianBlur(img1, (k_size, k_size), sigma)
        # 二值化边缘
        LowTh = 55
        ratio = 4
        cannyRet1 = mycanny(img1, LowTh, ratio, flag=3)
        # 显示
        cv.imshow(f'img{i+1}', img1)
        cv.imshow(f'cannyRet{i+1}', cannyRet1)
        cv.resizeWindow(f'img{i+1}', 350, 350)
        cv.resizeWindow(f'cannyRet{i+1}', 350, 350)
        cv.imwrite(f'canny{i+1}.jpg', cannyRet1, [cv.IMWRITE_JPEG_QUALITY, 100])

    # cv2官方Canny边缘检测函数
    # 二值化边缘
    LowTh = 55
    ratio = 4
    img1 = cv.imread(IMG1, cv.IMREAD_GRAYSCALE)
    cannyRet = cv.Canny(img1, LowTh, LowTh * ratio)
    cv.imshow('cannyRet', cannyRet)
    cv.resizeWindow('cannyRet', 350, 350)

    cv.waitKey(0)
    cv.destroyAllWindows()
