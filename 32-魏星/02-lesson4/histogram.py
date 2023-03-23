import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
直方图
反应了图像中的灰度分布规律，描述了每个灰度有多个像素个数；
图像直方图包括灰度直方图、颜色直方图；

绘制直方图可使用自定义方法、也可以使用函数calcHist
函数原型：calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=None)
images：图像矩阵，例如：[image]
channels：通道数，例如：0
mask：掩膜，一般为：None
histSize：直方图大小，一般等于灰度级数
ranges：横轴范围

'''

'''
自定义方法，计算灰度图像直方图
定义一个256长度的数组，用于存储每个像素(0-255)的个数
然后画直方图,x轴为0-255，y轴为该数组

'''
def histogram_gray_div(img):
    arr_x = np.array(range(256))
    arr = np.zeros((1, 256))

    w, h, c = img.shape
    for i in range(c):
        for x in range(w):
            for y in range(h):
                arr[0][img[x, y, i]] += 1

    plt.figure()
    # plt.bar(arr_x, arr[0])
    plt.plot(arr_x, arr[0])
    plt.show()

# 自定义计算图像彩色直方图
def histogram_color_div(img):
    w, h, c = img.shape

    arr_x = np.array(range(256))
    arr = np.zeros((c,256))
    colors=["b","g","r"]

    for i in range(c):
        for x in range(w):
            for y in range(h):
                arr[i][img[x, y, i]] += 1

    plt.figure()
    for i in range(c):
        # plt.bar(arr_x, arr[i])
        plt.plot(arr_x,arr[i],colors[i])
    plt.show()


img = cv2.imread("dog2.jpg")
histogram_color_div(img)

# img = cv2.imread("dog2.jpg")
# histogram_gray_div(img)
# img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 灰度图像的直方图，方法一
# plt.figure()
# plt.hist(img_gray.ravel(), 256)
# plt.show()

# 使用calcHist计算灰度图像
# hist = cv2.calcHist([img_gray],[0],None,[256],[0,256])
# plt.figure()#新建一个图像
# plt.title("Grayscale Histogram")
# plt.xlabel("Bins")#X轴标签
# plt.ylabel("# of Pixels")#Y轴标签
# plt.plot(hist)
# plt.xlim([0,256])#设置x坐标轴范围
# plt.show()


# 彩色图像直方图
# chans = cv2.split(img)
# colors = ("b","g","r")
# plt.figure()
# plt.title("Flattened Color Histogram")
# plt.xlabel("Bins")
# plt.ylabel("# of Pixels")
#
# for (chan,color) in zip(chans,colors):
#     hist = cv2.calcHist([chan],[0],None,[256],[0,256])
#     plt.plot(hist,color = color)
#     plt.xlim([0,256])
# plt.show()







