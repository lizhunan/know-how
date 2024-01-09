
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

### 基于频域的低/高通滤波
## 
## 1. 实现一维信号的傅里叶变换；
## 2. 实现图像的二维傅里叶变换；
## 3. 实现理想滤波器；
## 4. 实现Butterworth滤波器。
## 


# 1. 实现一维信号的傅里叶变换

def show(ori, fft, filename, sampling_period=5):
    n = len(ori)
    interval = sampling_period / n
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, sampling_period, interval), ori, 'black')
    plt.xlabel('time')
    plt.title('ori')

    plt.subplot(2,1,2)
    frequency = np.arange(n / 2) / (n * interval)
    nfft = abs(fft[range(int(n / 2))] / n )
    plt.plot(frequency, nfft, 'red')
    plt.xlabel('Hz')
    plt.title('fft')
    plt.savefig(f'result/{filename}')

# 生成频率为1(角速度为 2 * pi)的单一正弦波
time = np.arange(0, 5, .005)
x = np.sin(2 * np.pi * 1 * time)
y = np.fft.fft(x)
show(x, y, 'sigle')

# 生成频率为1(角速度为 2 * pi)的叠加正弦波
x2 = np.sin(2 * np.pi * 10 * time)
x3 = np.sin(2 * np.pi * 40 * time)
x4 = np.sin(2 * np.pi * 50 * time)
x += x2 + x3 + x4
y = np.fft.fft(x)
show(x, y, 'superposition')

#### 从图中可以看出，这三个正弦波形进行叠加得出的图像都具备了各自波形的特点，
#### 当进行傅里叶变化时，每个基函数都是一个单频率谐波。

# 2. 实现图像的二维傅里叶变换

img = cv.imread('asset/lena_color_512.tif')

# 转换成灰度图，并使用灰度直方图展示出来
# 发现灰度值集中在右侧
img_gray = cv.cvtColor(img, cv.COLOR_RGBA2GRAY)

# 快速傅里叶变换算法得到频率分布
fft = np.fft.fft2(img_gray)
# 默认结果中心点位置是在左上角，转移到中间位置
fshift = np.fft.fftshift(fft) 
# fft 结果是复数，求绝对值结果才是振幅
fimg = np.log(np.abs(fft)) 
fsimg = np.log(np.abs(fshift)) 

plt.figure(figsize=(12, 12))
plt.subplot(3, 1, 1)
plt.title('gray')
plt.imshow(img_gray, cmap='gray')
plt.subplot(3, 1, 2)
plt.title('fft')
plt.imshow(fimg, cmap='gray')
plt.subplot(3, 1, 3)
plt.title('fshift')
plt.imshow(fsimg, cmap='gray')
plt.tight_layout()
plt.savefig('result/fft.png')

# 3. 实现理想滤波器

def ideal_filter(img, D=50, mode='L'):
    # 傅里叶变换
    f1=np.fft.fft2(img)
    f1_shift=np.fft.fftshift(f1)

    # 实现理想滤波器
    rows,cols=img.shape
    crow,ccol=int(rows/2),int(cols/2) # 计算频谱中心
    if mode == 'L':
        mask=np.zeros((rows,cols),np.uint8) # 生成rows行cols的矩阵，数据格式为uint8
    elif mode == 'H':
        mask=np.ones((rows,cols),np.uint8)
    for i in range(rows):
        for j in range(cols):
            if np.sqrt(i*i+j*j)<=D:
                if mode == 'L':
                    # 将距离频谱中心小于D的部分低通信息 设置为1，属于低通滤波
                    mask[crow - D:crow + D, ccol - D:ccol + D] = 1
                elif mode == 'H':
                    mask[crow - D:crow + D, ccol - D:ccol + D] = 0
    m=f1_shift*mask

    # 傅里叶逆变换
    f_ishift = np.fft.ifftshift(m)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = (img_back-np.amin(img_back))/(np.amax(img_back)-np.amin(img_back))

    return img_back, m

# 4. 实现Butterworth滤波器

def butterworth_filter(img, D=50, n=1, mode='L'):
    # 傅里叶变换
    f1=np.fft.fft2(img)
    f1_shift=np.fft.fftshift(f1)
    s1 = np.log(np.abs(fshift))

    # 实现Butterworth滤波器
    def make_transform_matrix(D):
        transfor_matrix = np.zeros(img.shape)
        center_point = tuple(map(lambda x: (x - 1) / 2, s1.shape))
        for i in range(transfor_matrix.shape[0]):
            for j in range(transfor_matrix.shape[1]):
                def cal_distance(pa, pb):
                    from math import sqrt
                    dis = sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
                    return dis

                dis = cal_distance(center_point, (i, j))
                transfor_matrix[i, j] = 1 / (1 + (dis / D) ** (2*n))
        return transfor_matrix

    d_matrix = make_transform_matrix(D)

    # 傅里叶逆变换
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * d_matrix))), d_matrix
    return img_back

ideal = ideal_filter(img_gray, mode='L')
butterworth = butterworth_filter(img_gray, mode='L')
fsimg = np.log(np.abs(ideal[1])) 

plt.figure(figsize=(12, 12))
plt.subplot(3, 1, 1)
plt.title('back img')
plt.imshow(ideal[0], cmap='gray')

plt.subplot(3, 1, 2)
plt.title('fsimg')
plt.imshow(fsimg, cmap='gray')

plt.subplot(3, 1, 3)
plt.title('butterworth')
plt.imshow(butterworth[0], cmap='gray')

plt.tight_layout()
plt.savefig('result/filter.png')