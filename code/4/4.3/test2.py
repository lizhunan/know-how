
import cv2 as cv
import matplotlib.pyplot as plt

### 基于直方图均衡的图像增强
##
## 思路：先把图片转换为灰度图，根据图像的灰度值分布进行直方图均衡
## 
## 使用库：matplotlib库：展示分布直方图；
#         hist()：根据数据源和像素级绘制直方图；
#         revel()：由于灰度图像是由一个二维数组组成，所以需要使用revel()函数进行转换
#         equalizeHist()：直方图均衡，
#                         可以自己写算法，但是opencv自带了一个库直接用就好
# 经过图像增强后，jetplane.tif图中的山峰和云层的轮廓更加清晰
###

img = cv.imread('asset/jetplane.tif')

# 转换成灰度图，并使用灰度直方图展示出来
# 发现灰度值集中在右侧
img_gray = cv.cvtColor(img, cv.COLOR_RGBA2GRAY)

plt.subplot(221)
plt.imshow(img_gray, cmap='gray')
plt.subplot(223)
plt.hist(img_gray.ravel(), 512)

equ = cv.equalizeHist(img_gray)

plt.subplot(222)
plt.imshow(equ, cmap='gray')
plt.subplot(224)
plt.hist(equ.ravel(), 512)
plt.savefig('result/equ.png')