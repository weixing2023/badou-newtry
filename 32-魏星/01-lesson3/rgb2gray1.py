import numpy as np
import cv2
import matplotlib.pyplot as plt

'''
灰度化
将rgb转换为gray，转换结果图片大小一致，即(w,h,c)->(w,h)
可使用浮点算法、平均值法、单通道法、整数方法、位移方法
步骤：
先创建一个和多通道图像w,h一致的单通道图片，
然后循环多通道的每个像素点位，使用指定算法，将该点位的多通道信息转换为单通道信息
最后生成单通道图片
'''

img = cv2.imread("dog1.jpg")
print(img.shape) #查看数字图像shape,结果为w,h,c，三通道图片
w,h = img.shape[:2] #取图像w,h
# print(w,h)
img_gray = np.zeros((w,h),img.dtype)
# print(img_gray)
for i in range(w):
    for j in range(h):
        last=img[i,j]
        # print(last)
        # img_gray[i,j] = int(np.sum(last)/3) #使用平均值法
        img_gray[i, j] = int(last[0]*0.11+last[1]*0.59+last[2]*0.3) #使用浮点算法 gray=R*0.3+g*0.59+b*0.11,cv2数字图像格式为BGR
# print(img_gray)
# cv2.imshow("image show gray", img_gray)

plt.subplot(121) #subplot(m,n,p)或者subplot(mnp)是将多个图画到一个平面上的工具。其中m表示行，n表示列，p表示从左到右从上到下的第p个位置
plt.imshow(img) #画原图

plt.subplot(122)
plt.imshow(img_gray) #画灰度图

plt.show()



