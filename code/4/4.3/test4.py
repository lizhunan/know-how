
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


### 基于空间域的高通滤波
##


def Robert(img):
    kernelX = np.array([[-1, 0], [0, 1]], dtype=np.uint)
    kernelY = np.array([[0, -1], [1, 0]], dtype=np.uint)
    
    x = cv.filter2D(img, cv.CV_16S, kernelX)
    y = cv.filter2D(img, cv.CV_16S, kernelY)

    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)
    
    return cv.addWeighted(absX, 0.5, absY, 0.5, 0)

def Prewitt(img):
    kernelX = np.array([[1,1,1],[0,0,0],[-1,-1,-1]],dtype=np.uint)
    kernelY = np.array([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=np.uint)

    x = cv.filter2D(img, cv.CV_16S, kernelX)
    y = cv.filter2D(img, cv.CV_16S, kernelY)

    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)
    
    return cv.addWeighted(absX, 0.5, absY, 0.5, 0)

def Sobel(img):
    kernelX = np.array([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=np.uint)
    kernelY = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=np.uint)

    x = cv.filter2D(img, cv.CV_16S, kernelX)
    y = cv.filter2D(img, cv.CV_16S, kernelY)

    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)
    
    return cv.addWeighted(absX, 0.5, absY, 0.5, 0)

def Laplacian(img):
    dst = cv.Laplacian(img, cv.CV_16S, ksize = 3)
    return cv.convertScaleAbs(dst)

def Canndy(img):
    return cv.Canny(img, 270, 300)

img = cv.imread('asset/saltpep_prob.tif')

# 转换成灰度图，并使用灰度直方图展示出来
# 发现灰度值集中在右侧
img_gray = cv.cvtColor(img, cv.COLOR_RGBA2GRAY)

# 高斯滤波s
gaussian_blur = cv.GaussianBlur(img_gray, (5, 5), 0)

plt.figure(figsize=(60, 60))

plt.subplot(6, 2, 1)
plt.imshow(img_gray, cmap='gray')
plt.title('gray')

plt.subplot(6, 2, 2)
plt.imshow(gaussian_blur, cmap='gray')
plt.title('gaussian blur')

plt.subplot(6, 2, 3)
img_sobel = Robert(img_gray)
plt.imshow(img_sobel, cmap='gray')
plt.title('Robert')

plt.subplot(6, 2, 4)
img_sobel = Robert(gaussian_blur)
plt.imshow(img_sobel, cmap='gray')
plt.title('Robert')

plt.subplot(6, 2, 5)
img_prewitt = Prewitt(img_gray)
plt.imshow(img_prewitt, cmap='gray')
plt.title('Prewitt')

plt.subplot(6, 2, 6)
img_prewitt = Prewitt(gaussian_blur)
plt.imshow(img_prewitt, cmap='gray')
plt.title('Prewitt')

plt.subplot(6, 2, 7)
img_sobel = Sobel(img_gray)
plt.imshow(img_sobel, cmap='gray')
plt.title('Sobel')

plt.subplot(6, 2, 8)
img_sobel = Sobel(gaussian_blur)
plt.imshow(img_sobel, cmap='gray')
plt.title('Sobel')

plt.subplot(6, 2, 9)
img_laplacian = Laplacian(img_gray)
plt.imshow(img_laplacian, cmap='gray')
plt.title('Laplacian')

plt.subplot(6, 2, 10)
img_laplacian = Laplacian(gaussian_blur)
plt.imshow(img_laplacian, cmap='gray')
plt.title('Laplacian')

plt.subplot(6, 2, 11)
img_canndy = Canndy(img_gray)
plt.imshow(img_canndy, cmap='gray')
plt.title('Canndy')

plt.subplot(6, 2, 12)
img_canndy = Canndy(gaussian_blur)
plt.imshow(img_canndy, cmap='gray')
plt.title('Canndy')

plt.tight_layout()
plt.savefig('result/sharpen.png')