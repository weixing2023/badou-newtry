import cv2
import numpy as np
from matplotlib import pyplot as plt


'''
直方图均衡化

满足两个条件：
1、不打乱原图像的亮度顺序，映射前后的亮、暗关系不能变
2、映射后取值在原有的范围内，比如(0-255)

'''

# 自定义计算彩色直方图均衡化
'''
步骤：
1、依次扫描原始灰度图像的每一个像素，计算出图像的灰度直方图H
2、计算灰度直方图的累加直方图
3、根据累加直方图和直方图均衡化原理得到输入与输出之间的映射关系
4、最后根据映射关系得到结果：dst(x,y)=H'(src(x,y))进行图像变换

'''
def histogram_equalization_color_div(img):
    w, h, c = img.shape
    result_img = np.zeros((w,h,c),img.dtype) #此处需要定义一个新的图像，或者深copy原图像，不得使用原来的img，会修改掉原图像

    arr = np.zeros((c, 256))
    for i in range(c):
        for x in range(w):
            for y in range(h):
                arr[i][img[x, y, i]] += 1

        sum = 0
        for j in range(256):
            sum += arr[i][j]
            arr[i][j] = int(sum*256/(w*h) - 1)

        for x in range(w):
            for y in range(h):
                result_img[x, y, i] = arr[i][img[x, y, i]]

    return result_img

def color_equalization(img):
    chans = cv2.split(img)
    colors = ("r", "g", "b")
    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])


img = cv2.imread("dog2.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result_img = histogram_equalization_color_div(img_rgb)

plt.figure()
plt.subplot(221)
plt.imshow(img_rgb)
plt.title("img_rgb figure")

plt.subplot(222)
plt.imshow(result_img)
plt.title("img_equalization figure")

plt.subplot(223)
color_equalization(img_rgb)
plt.title("img_rgb histogram figure")

plt.subplot(224)
color_equalization(result_img)
plt.title("img_equalization histogram figure")

plt.show()


# 获取灰度图像
# img = cv2.imread("dog2.jpg", 1)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # 灰度图像直方图均衡化
# dst = cv2.equalizeHist(gray)
# # 直方图
# hist = cv2.calcHist([dst],[0],None,[256],[0,256])
#
# plt.figure()
# plt.subplot(221)
# plt.imshow(gray)
#
# plt.subplot(222)
# plt.imshow(dst)
#
# plt.subplot(223)
# plt.hist(gray.ravel(), 256)
#
# plt.subplot(224)
# plt.hist(dst.ravel(), 256)
# plt.show()


# 彩色图像直方图均衡化
# img = cv2.imread("dog1.jpg")
# cv2.imshow("src", img)
#
# # 彩色图像均衡化,需要分解通道 对每一个通道均衡化
# (b, g, r) = cv2.split(img)
# bH = cv2.equalizeHist(b)
# gH = cv2.equalizeHist(g)
# rH = cv2.equalizeHist(r)
# # 合并每一个通道
# result = cv2.merge((bH, gH, rH))
# cv2.imshow("dst_rgb", result)
#
# cv2.waitKey(0)






