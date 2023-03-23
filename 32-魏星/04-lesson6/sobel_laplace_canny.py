import cv2
import matplotlib.pyplot as plt


img = cv2.imread("dog2.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# sobel_x
img_sobelx = cv2.Sobel(img_gray, cv2.CV_64F,1,0,ksize=3)
abs_sobelx = cv2.convertScaleAbs(img_sobelx) #取绝对值

# sobel_y
img_sobely = cv2.Sobel(img_gray, cv2.CV_64F,0,1,ksize=3)
abs_sobely = cv2.convertScaleAbs(img_sobely)

# laplace
img_laplace = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=3)
abs_laplace = cv2.convertScaleAbs(img_laplace)

# canny
'''
edge = cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]])
image：要检测的图像
threshold1：阈值1（最小值）
threshold2：阈值2（最大值），使用此参数进行明显的边缘检测
edges：图像边缘信息
apertureSize：sobel算子（卷积核）大小
L2gradient ：布尔值。
True： 使用更精确的L2范数进行计算（即两个方向的导数的平方和再开方）
False：使用L1范数（直接将两个方向导数的绝对值相加）
'''
img_canny = cv2.Canny(img_gray, 20, 240, apertureSize=3)

plt.figure()
plt.subplot(341)
plt.title("original figure")
plt.imshow(img_rgb)

plt.subplot(342)
plt.title("img_gray")
plt.imshow(img_gray)

plt.subplot(343)
plt.title("sobel_x")
plt.imshow(img_sobelx)

plt.subplot(344)
plt.title("abs sobel_x")
plt.imshow(abs_sobelx)

plt.subplot(345)
plt.title("sobel_y")
plt.imshow(img_sobely)

plt.subplot(346)
plt.title("abs sobel_y")
plt.imshow(abs_sobely)

plt.subplot(347)
plt.title("img_laplace")
plt.imshow(img_laplace)

plt.subplot(348)
plt.title("abs laplace")
plt.imshow(abs_laplace)

plt.subplot(349)
plt.title("canny")
plt.imshow(img_canny)
plt.show()




