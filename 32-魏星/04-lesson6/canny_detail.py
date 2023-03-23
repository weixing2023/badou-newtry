import cv2

'''
canny实现步骤：
1、灰度化
2、高斯滤波（降噪）
3、检测图像边缘（soble）
4、像素点非最大值抑制(NMS)
5、双阈值检测连接边缘

'''

# 灰度化
img = cv2.imread("dog1.jpg")
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)





