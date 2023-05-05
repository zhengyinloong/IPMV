# -*- coding:utf-8 -*-
# demo.py
# zhengyinloong
# 2023/04/28 23:53
"""
Hough检测
Harris角点检测
OTSU法实现图像分割
"""
from mycv import *

# IMG = "./10.jpg"
IMG = "./kunkun.jpg"

img = cv.imread(IMG)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# Canny 边缘检测
Edge = edge_detecting(gray,sigma=30,lowTh=2)
# Hough 直线检测
lines = get_lines(Edge,70)
# 画线
Hough = img.copy()
Hough = draw_lines(Hough, lines)
# Harris 角点检测
# 得到角点
# corners = myharris(gray)
corners = myharris(Edge)
Harris = img.copy()
Harris = draw_corners(corners, Harris)
# OTSU 语义分割
# 得到最佳阈值
threshold = otsu_threshold(gray)
# 分割
Otsu = np.zeros_like(gray)
Otsu[gray>threshold]=255
# 显示
cv.imshow('Edge', Edge)
cv.imshow('Hough', Hough)
cv.imshow('Harris', Harris)
cv.imshow('Otsu', Otsu)
# 保存
cv.imwrite('edge.jpg', Edge)
cv.imwrite('hough.jpg', Hough)
cv.imwrite('Harris.jpg', Harris)
cv.imwrite('Otsu.jpg', Otsu)


# Wait until a key pressed
cv.waitKey(0)

# Destroy all the windows opened before
cv.destroyAllWindows()