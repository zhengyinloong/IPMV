# -*- coding:utf-8 -*-
# demo2-hough.py
# zhengyinloong
# 2023/04/28 07:29
"""
Hough直线检测
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

    # Wait until a key pressed
    cv.waitKey(0)

    # Destroy all the windows opened before
    cv.destroyAllWindows()



