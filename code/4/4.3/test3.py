
import cv2 as cv
from skimage import img_as_ubyte  
from skimage import util
import matplotlib.pyplot as plt

### 基于空间域的低通滤波
##
## 给出均值滤波、中值滤波和高斯滤波示例代码并展示效果
## 在示例代码中，首先根据原图创建了高斯噪声（gauss_noisy）和椒盐噪声（sp_noisy） 
## 使用均值滤波、中值滤波和高斯滤波分别对这两种噪声进行去噪并作对比
##
## 使用库：matplotlib库：展示分布直方图；
#         skimage库：主要用于给原图增加噪声
#         blur()：均值滤波
#         medianBlur()：中值滤波
#         GaussianBlur()：高斯滤波
# 经过图像增强后，lena_color_512.tif图中噪声均被消除
# 但是根据使用方法的不同，效果差异较大
# 其中，对于椒盐噪声，中值滤波效果最好
# 对于高斯噪声，高斯滤波效果最好
# 中值滤波由于卷积核的内在缺陷，对噪声处理不佳
# 并且会对图像边界进行模糊
###

img = cv.imread('asset/lena_color_512.tif')

# 转换成灰度图，并使用灰度直方图展示出来
# 发现灰度值集中在右侧
img_gray = cv.cvtColor(img, cv.COLOR_RGBA2GRAY)

plt.figure(figsize=(12, 12))

plt.subplot(4, 3, 1)
plt.imshow(img_gray, cmap='gray')
plt.title('gray')

plt.subplot(4, 3, 2)
gauss_noisy = util.random_noise(img_gray, mode='gaussian', mean=0.1, var=0.01)
gauss_noisy = img_as_ubyte(gauss_noisy)
plt.title('gauss_noisy')
plt.imshow(gauss_noisy, cmap='gray')

plt.subplot(4, 3, 3)
sp_noisy = util.random_noise(img_gray, mode='s&p', amount=0.05)
sp_noisy = img_as_ubyte(sp_noisy)
plt.title('salt&pepper_noisy')
plt.imshow(sp_noisy, cmap='gray')

plt.subplot(4, 3, 5)
blur = cv.blur(gauss_noisy, (5,5))
plt.title('blur')
plt.imshow(blur, cmap='gray')
plt.subplot(4, 3, 6)
blur = cv.blur(sp_noisy, (5,5))
plt.title('blur')
plt.imshow(blur, cmap='gray')

plt.subplot(4, 3, 8)
med_blur = cv.medianBlur(gauss_noisy, 5)
plt.title('median blur')
plt.imshow(med_blur, cmap='gray')
plt.subplot(4, 3, 9)
med_blur = cv.medianBlur(sp_noisy, 5)
plt.title('median blur')
plt.imshow(med_blur, cmap='gray')

plt.subplot(4, 3, 11)
gauss = cv.GaussianBlur(gauss_noisy, (5,5), 0)
plt.title('gauss blur')
plt.imshow(gauss, cmap='gray')
plt.subplot(4, 3, 12)
gauss = cv.GaussianBlur(sp_noisy, (5,5), 0)
plt.title('gauss blur')
plt.imshow(gauss, cmap='gray')

plt.tight_layout()
plt.savefig('result/blur.png')