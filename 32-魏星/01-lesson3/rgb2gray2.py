import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

'''
使用库方法实现灰度化
'''

img = cv2.imread("dog1.jpg")
img_normal = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray1 = rgb2gray(img)
img_gray2 = rgb2gray(img_normal)

plt.subplot(221)
plt.imshow(img) #画cv2读出的原图

plt.subplot(222)
plt.imshow(img_normal) #画bgr转rgb后图像

plt.subplot(223)
plt.imshow(img_gray1) #画灰度图

plt.subplot(224)
plt.imshow(img_gray2) #画灰度图

plt.show()



