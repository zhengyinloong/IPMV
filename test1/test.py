import numpy as np
import cv2 as cv
# a = np.array([[1,2,3],[1,2,3],[1,2,3]])
#
# k= a==1
# a[k]=4
# k= a==2
# a[k]=5
# k= a==3
# a[k]=6
# print(a[0:2][0:2])
# print(np.round(4.4))
# img5 = np.zeros((2,3),dtype=np.uint8)
# print(img5)
# i = 2
# if 1<i<3:
#     print("a")
#
# imgFile = 'img.jpg'
# Img = cv.imread(imgFile, cv.IMREAD_GRAYSCALE)
# H, W = np.shape(Img)
# print(W, H)

# 缩放
# w = int(W / 4)
# h = int(H / 4)
# img1=cv.resize(Img,(w,h))
# Img2=cv.GaussianBlur(img1,(3,3),1)
# cv.imshow('Img2',Img2)
# cv.waitKey(0)
# 1/sqrt(2*pi)*exp(-(power(x,2)+power(y,2))/(2*power(sigma,2)))
def gauss(sigma,d):
    g = np.zeros((d,d),dtype=np.float64)
    # o = np.median(range(1,d+1))
    o = (d+1)/2
    for i in range(d):
        for j in range(d):
            g[i,j] = 1/np.sqrt(2*np.pi*np.power(sigma,2))\
                     *np.exp(-(np.power(i+1-o,2)+np.power(j+1-o,2))/(2*np.power(sigma,2)))
    g = g/np.sum(g)
    return g
print(gauss(1.5,3))
