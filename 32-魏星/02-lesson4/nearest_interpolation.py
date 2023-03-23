import numpy as np
import cv2

'''
上采样原理是内插值，常见的上采样方法：插值、反卷积（转置卷积）、反池化

1、最邻近插值
图像放大(上采样)
'''

# 最邻近插值算法，img：原图像，times：缩放的倍数
def nearest_interpolation(img, times):
    # if times<1 or times==1:
    #     return img #暂时返回输入图像

    if times < 0 or times == 0:
        return img

    w,h,c=img.shape #原图像尺寸和通道值
    rw = int(w*times) #放大后图像的w
    rh = int(h*times) #放大后图像的h
    result_img = np.zeros((rw,rh,c), img.dtype) #首先生成一个放大的空的图像
    for i in range(rw):
        for j in range(rh):
            x = int(i/times + 0.5)
            y = int(j/times + 0.5)
            # 图像放大，如果选取的像素点位index超出了原图像的尺寸，则该点位像素值设置255
            if x<w and y<h:
                result_img[i, j] = img[x, y]
            else:
                result_img[i, j] = 255
    return result_img

img = cv2.imread("dog2.jpg")
print("--normal figure's shape ", img.shape)
# print(img)
large_img = nearest_interpolation(img, 1.6)
print("--large figure's shape ", large_img.shape)
# print(large_img)

cv2.imshow("normal img", img)
cv2.imshow("nearest interpolation result figure", large_img)
cv2.waitKey(0)




