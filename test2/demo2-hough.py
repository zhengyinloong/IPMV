# -*- coding:utf-8 -*-
# demo2-hough.py
# zhengyinloong
# 2023/04/28 07:29
"""
霍夫直线检测
"""
from mycv import *


if __name__=="__main__":

    IMG1 = "./10.jpg"
    IMG2 = "./11.jpg"
    IMG3 = "./12.jpg"
    IMGS = [IMG1,IMG2,IMG3]

    for i,IMG in enumerate(IMGS):
        img = cv.imread(IMG, cv.IMREAD_GRAYSCALE)
        # Canny 边缘检测
        edge = edge_detecting(img)
        # Hough直线检测
        lines = get_lines(edge)
        # 画线
        img = cv.imread(IMG)
        hough = draw_lines(img, lines)
        # 显示
        cv.imshow(f'edge{i+1}', edge)
        cv.imshow(f'hough{i+1}', hough)
        cv.resizeWindow(f'edge{i+1}', 350, 350)
        cv.resizeWindow(f'hough{i+1}', 350, 350)
        # 保存
        cv.imwrite(f'edge{i+1}.jpg', edge, [cv.IMWRITE_JPEG_QUALITY, 100])
        cv.imwrite(f'hough{i+1}.jpg', hough, [cv.IMWRITE_JPEG_QUALITY, 100])


    # img1 = cv.imread(IMG1,cv.IMREAD_GRAYSCALE)
    # img2 = cv.imread(IMG2,cv.IMREAD_GRAYSCALE)
    # img3 = cv.imread(IMG3,cv.IMREAD_GRAYSCALE)
    # edge1 = edge_detecting(img1)
    # edge2 = edge_detecting(img2)
    # edge3 = edge_detecting(img3)
    #
    # lines1=get_lines(edge1)
    # lines2=get_lines(edge2)
    # lines3=get_lines(edge3)
    #
    # # 画线
    # img1 = cv.imread(IMG1)
    # img2 = cv.imread(IMG2)
    # img3 = cv.imread(IMG3)
    # hough1 = draw_lines(img1,lines1)
    # hough2 = draw_lines(img2,lines2)
    # hough3 = draw_lines(img3,lines3)
    #
    #
    # # 显示
    # cv.imshow('edge1',edge1)
    # cv.imshow('edge2',edge2)
    # cv.imshow('edge3',edge3)
    # cv.imshow('hough1',hough1)
    # cv.imshow('hough2',hough2)
    # cv.imshow('hough3',hough3)

    # Wait until a key pressed
    cv.waitKey(0)

    # Destroy all the windows opened before
    cv.destroyAllWindows()



