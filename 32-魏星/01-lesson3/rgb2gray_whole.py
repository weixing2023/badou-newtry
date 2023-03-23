import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray


# 自定义方法灰度化
def rgb2gray_div(img):
    w, h = img.shape[:2]  # 取图像w,h
    # print(w,h)
    img_gray = np.zeros((w, h), img.dtype)
    # print(img_gray)
    for i in range(w):
        for j in range(h):
            last = img[i, j]
            # print(last)
            # img_gray[i,j] = int(np.sum(last)/3) #使用平均值法
            img_gray[i, j] = int(
                last[0] * 0.11 + last[1] * 0.59 + last[2] * 0.3)  # 使用浮点算法 gray=R*0.3+g*0.59+b*0.11,cv2数字图像格式为BGR
    # print(img_gray)
    return img_gray

# 二值化
def rgb2gray_binaryzation(img):
    w, h = img.shape[:2]  # 取图像w,h
    # print(w,h)
    img_gray = np.zeros((w, h), img.dtype)
    # print(img_gray)
    for i in range(w):
        for j in range(h):
            last = img[i, j]
            gray_num = last[0] * 0.11 + last[1] * 0.59 + last[2] * 0.3  # 使用浮点算法 gray=R*0.3+g*0.59+b*0.11,cv2数字图像格式为BGR
            if gray_num <= 127.5:
                img_gray[i, j] = 0
            else:
                img_gray[i, j] = 1
    # print(img_gray)
    return img_gray

# 二值化
def rgb2gray_binary(img):
    img_gray = rgb2gray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_binary = np.where(img_gray >= 127.5, 1, 0)
    return img_binary


img = cv2.imread("dog2.jpg")
# cv2，bgr转rgb
img_gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 自定义灰度化
img_gray2 = rgb2gray_div(img)
# 调用库方法rgb2gray
img_gray3 = rgb2gray(img_gray1)
# 二值化
img_gray4 = rgb2gray_binaryzation(img)


plt.subplot(231)
plt.title("cv2-bgr figure")
plt.imshow(img) #画原图

plt.subplot(232)
plt.title("bgr2rgb figure")
plt.imshow(img_gray1) #画bgr转rgb后图像

plt.subplot(233)
plt.title("rgb2gray_div figure")
plt.imshow(img_gray2) #画自定义灰度化方法图像

plt.subplot(234)
plt.title("rgb2gray_libary figure")
plt.imshow(img_gray3) #画使用已有库方法灰度化后的图像

plt.subplot(235)
plt.title("binaryzation figure")
plt.imshow(img_gray4) #画二值化图

plt.show()






