
import cv2 as cv
import matplotlib.pyplot as plt

### 将图像转为灰度直方图
##
## 思路：先把图片转换为灰度图，然后根据灰度值的分布来绘制直方图
## 使用库：matplotlib库：展示分布直方图；
#         hist()：根据数据源和像素级绘制直方图；
#         revel()：由于灰度图像是由一个二维数组组成，所以需要使用revel()函数进行转换
###

img = cv.imread('asset/jetplane.tif')

# 转换成灰度图
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#获取直方图，由于灰度图img2是二维数组，需转换成一维数组
plt.subplot(211)
plt.imshow(img, cmap='gray')
plt.subplot(212)
plt.hist(img_gray.ravel(), 512)
plt.savefig('result/hist.png')