import numpy as np
import matplotlib.pyplot as plt
import cv2

'''
二值化(像素值只有0或1)
现象图像灰度化，计算灰度化后的像素的值，
如<0.5则置为0，否则为1
'''

img = cv2.imread("dog1.jpg")
print(img.shape) #查看数字图像shape,结果为w,h,c，三通道图片
w,h = img.shape[:2] #取图像w,h
img_gray = np.zeros((w,h),img.dtype)
for i in range(w):
    for j in range(h):
        last=img[i,j]
        gray_num=last[0]*0.11+last[1]*0.59+last[2]*0.3 #使用浮点算法 gray=R*0.3+g*0.59+b*0.11,cv2数字图像格式为BGR
        if gray_num <= 127.5:
            img_gray[i,j]=0
        else:
            img_gray[i,j]=1
# print(img_gray)
# cv2.imshow("image show gray", img_gray)

plt.subplot(121)
plt.imshow(img) #画cv2读出的原图

plt.subplot(122)
plt.imshow(img_gray) #画二值图

plt.show()
