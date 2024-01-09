
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

### 形态学处理（简单运算）
## 
## 1. 膨胀
## 2. 腐蚀
## 3. 开运算
## 4. 闭运算
## 

img = cv.imread('asset/saltpep_prob.tif', 0)

ret2, th2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

# 用OpenCV中的getStructuringElement()函数定义结构元素
kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))

eroded = cv.erode(th2, kernel)
dilate = cv.dilate(th2, kernel)
open = cv.morphologyEx(th2, cv.MORPH_OPEN, kernel) 
close = cv.morphologyEx(th2, cv.MORPH_CLOSE, kernel)

plt.figure(figsize=(12, 12))
plt.subplot(5, 1, 1)
plt.title('img')
plt.imshow(img, cmap='gray')
plt.subplot(5, 1, 2)
plt.title('eroded')
plt.imshow(eroded, cmap='gray')
plt.subplot(5, 1, 3)
plt.title('dilate')
plt.imshow(dilate, cmap='gray')
plt.subplot(5, 1, 4)
plt.title('open')
plt.imshow(open, cmap='gray')
plt.subplot(5, 1, 5)
plt.title('close')
plt.imshow(close, cmap='gray')

plt.tight_layout()
plt.savefig('result/morphology.png')