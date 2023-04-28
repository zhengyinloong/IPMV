# -*- coding:utf-8 -*-
# mycv.py
# zhengyinloong
# 2023/04/27 09:07
"""
    图像处理自定义系列函数
    Grads
    Canny
    Hough
    ...
"""
import cv2 as cv
import numpy as np

""" 梯度算子 """

# Sobel算子
# Sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
# Sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
# Scharr算子
Sobel_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
Sobel_y = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])


def myconv(img, myfilter):
    # 该函数借鉴自CSDN
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        img: numpy array of shape (Hi, Wi).
        myfilter: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = img.shape
    Hk, Wk = myfilter.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0, pad_width0), (pad_width1, pad_width1))
    padded = np.pad(img, pad_width, mode='edge')

    ### YOUR CODE HERE
    x = Hk // 2
    y = Wk // 2
    # 横向遍历卷积后的图像
    for i in range(pad_width0, Hi - pad_width0):
        # 纵向遍历卷积后的图像
        for j in range(pad_width1, Wi - pad_width1):
            split_img = img[i - pad_width0:i + pad_width0 + 1, j - pad_width1:j + pad_width1 + 1]
            # 对应元素相乘
            out[i, j] = np.sum(np.multiply(split_img, myfilter))
    # out = (out-out.min()) * (1/(out.max()-out.min()) * 255).astype('uint8')

    return out


def mygrads(img):
    """
    计算梯度,这里暂时使用的梯度算子是 Scharr 算子
    :param img:
    :return: [grads, theta]
    """
    gx = myconv(img, Sobel_x)
    gy = myconv(img, Sobel_y)
    grads = np.sqrt(np.square(gx) + np.square(gy))

    theta = np.arctan2(gy, gx) * 180 / np.pi

    return [grads, theta]


def mycanny(img, lowTh=None, ratio=None, flag=None):
    """
    canny edge detection
    :param img: gray image matrix
    :param lowTh: Low threshold
    :param ratio: higTh = ratio*lowTh
    :param flag: 0 - grads,
                 1 - sector,
                 2 - canny2,
                 3 - bins,
                 None - all [grads,theta,sector,canny1,canny2,bins]
    :return: img after canny
    """
    # img=np.double(img)
    [row, col] = np.shape(img)
    # 梯度大小，方向，区域矩阵
    # grads = np.zeros((row, col))
    # theta = np.zeros((row, col))
    [grads, theta] = mygrads(img)
    sector = np.zeros((row, col))
    # 非极大值抑制
    canny1 = np.zeros((row, col))
    # 双阈值检测和连接
    canny2 = np.zeros((row, col))
    # 二值化
    bins = np.zeros((row, col))
    for i in range(row - 1):
        for j in range(col - 1):
            if -67.5 < theta[i, j] <= -22.5 or 112.5 <= theta[i, j] < 157.5:
                sector[i, j] = 0
            elif 67.5 <= theta[i, j] < 112.5 or -112.5 < theta[i, j] <= -67.5:
                sector[i, j] = 1
            elif 22.5 <= theta[i, j] < 67.5 or -157.5 < theta[i, j] <= -112.5:
                sector[i, j] = 2
            else:
                sector[i, j] = 3
    # 非极大值抑制
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            if 0 == sector[i, j]:  # 右上 - 左下
                if (grads[i, j] > grads[i + 1, j - 1]) and (grads[i, j] > grads[i - 1, j + 1]):
                    canny1[i, j] = grads[i, j]
                else:
                    canny1[i, j] = 0
            elif 1 == sector[i, j]:  # 竖直方向
                if (grads[i, j] > grads[i - 1, j]) and (grads[i, j] > grads[i + 1, j]):
                    canny1[i, j] = grads[i, j]
                else:
                    canny1[i, j] = 0
            elif 2 == sector[i, j]:  # 左上-右下
                if (grads[i, j] > grads[i - 1, j - 1]) and (grads[i, j] > grads[i + 1, j + 1]):
                    canny1[i, j] = grads[i, j]
                else:
                    canny1[i, j] = 0
            elif 3 == sector[i, j]:  # 横方向
                if (grads[i, j] > grads[i, j - 1]) and (grads[i, j] > grads[i, j + 1]):
                    canny1[i, j] = grads[i, j]
                else:
                    canny1[i, j] = 0
    # 双阈值检测
    # 自动生成低阈值和高阈值(比率)，该方法由试验得来，有待进一步检验
    if lowTh is None:
        lowTh = np.mean(canny1)
        print("low threshold = ", lowTh)
    if ratio is None:
        ratio = np.max(canny1) / np.mean(canny1) * 0.6
        print("high threshold = ", ratio * lowTh)
    higTh = ratio * lowTh  # 高阈值
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            if canny1[i, j] < lowTh:
                canny2[i, j] = 0
                bins[i, j] = 0
                continue
            elif canny1[i, j] > higTh:
                canny2[i, j] = canny1[i, j]
                # bins[i,j] = 1
                bins[i, j] = 255
                continue
            else:
                # 8邻域连接,介于之间的看其8领域有没有高于高阈值的，有则可以为边缘
                # tem = [canny1[i-1,j-1], canny1[i-1,j], canny1[i-1,j+1],
                #        canny1[i,j-1],    canny1[i,j],   canny1[i,j+1],
                #        canny1[i+1,j-1], canny1[i+1,j], canny1[i+1,j+1]]
                tem = canny1[i - 1:i + 2, j - 1:j + 2]
                temMax = np.max(tem)
                if temMax > higTh:
                    canny2[i, j] = temMax
                    # bins[i,j]=1
                    bins[i, j] = 255
                    continue
                else:
                    canny2[i, j] = 0
                    bins[i, j] = 0
                    continue

    canny1 = np.round(canny1)
    canny2 = np.round(canny2)
    bins = np.round(bins)

    # 返回值分配
    if flag == 0:
        return grads
    elif flag == 1:
        return sector
    elif flag == 2:
        return canny2
    elif flag == 3:
        return bins
    else:
        return [grads, theta, sector, canny1, canny2, bins]


def edge_detecting(img):
    # Gray
    # 高斯降噪
    sigma = 10
    k_size = 3
    img = cv.GaussianBlur(img, (k_size, k_size), sigma)
    # canny检测
    edges = mycanny(img, flag=3)
    return edges


def get_lines(edge):
    dtheta = 1
    # rho能达到的最大值，rho = x*np.cos(theta)+y*np.sin(theta)，取整
    rhoMax = round(np.sqrt(np.square(edge.shape[0]) + np.square(edge.shape[1])))
    Q = np.zeros((rhoMax, 180))
    # 得到Canny检测到的边缘点坐标
    edge_y, edge_x = np.where(edge == 255)

    for x, y in zip(edge_x, edge_y):
        for i in range(0, 180, dtheta):
            theta = i * np.pi / 180
            rho = round(x * np.cos(theta) + y * np.sin(theta))
            Q[rho, i] += 1
    """
    这里使用的方法是：Q.ravel()将Q横向展成一维数组X，然后npargsort()将X中的元素从小到大排列，提取其对应的index(索引)并输出到_X
    然后[::-1]将_X中的索引倒序输出,[:N]取前N个索引，由此得到前N大的值的索引值
    """
    N = 20
    X = Q.ravel()
    _X = np.argsort(X)
    ind_x = _X[::-1][:N]
    ind_y = ind_x.copy()
    # 得到相应的rho和theta,
    thetas = ind_x % 180
    rhos = ind_y // 180
    # print(thetas,rhos)
    # print(Q[rhos,thetas])
    # _Q = np.zeros_like(Q)
    # _Q[rhos,thetas] = 255
    # cv.imshow('Q',Q)
    # cv.imshow('_Q',_Q)

    return [thetas, rhos]


def draw_lines(img, lines):
    H, W, C = img.shape

    [thetas, rhos] = lines
    for theta, rho in zip(thetas, rhos):
        t = np.pi / 180. * theta
        for x in range(W):
            if np.sin(t) != 0:
                y = - (np.cos(t) / np.sin(t)) * x + rho / np.sin(t)
                y = int(y)
                if y >= H or y < 0:
                    continue
                img[y, x] = [0, 0, 255]
        for y in range(H):
            if np.cos(t) != 0:
                x = - (np.sin(t) / np.cos(t)) * y + rho / np.cos(t)
                x = int(x)
                if x >= W or x < 0:
                    continue
                img[y, x] = [0, 0, 255]
    return img
    pass

# ========================= THE END ======================== #
