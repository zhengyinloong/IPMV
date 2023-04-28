# -*- coding:utf-8 -*-
# test.py
# zhengyinloong
# 2023/04/26 17:26
import numpy as np

a = np.zeros((5,5))
a[2,2:4]=5
i=3
j=3
b= a[i-1:i+2,j-1:j+2]
# print(np.max(a))
print(np.arctan2(10000000,1))