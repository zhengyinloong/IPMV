# -*- coding:utf-8 -*-
# test.py
# zhengyinloong
# 2023/05/03 20:56
import numpy as np

# Prewitt算子
Prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
Prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
# Sobel算子
Sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
Sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
# Scharr算子
Scharr_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
Scharr_y = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])

G_optr_x = [Prewitt_x,Sobel_x,Scharr_x]
G_optr_y = [Prewitt_y,Sobel_y,Scharr_y]

print(Scharr_x,
      Scharr_y,Scharr_x*np.sin(Scharr_y),sep='\n')