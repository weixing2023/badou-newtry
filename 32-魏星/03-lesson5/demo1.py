
import numpy as np
import matplotlib.pyplot as plt
import cv2


img = cv2.imread("dog1.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img_gray.shape)
w,h = img_gray.shape
for i in range(w):
    for j in range(h):
        if img_gray[i,j] != 255:
            img_gray[i,j] = 0

kernel_gx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
print(kernel_gx)
kernel_gy = kernel_gx.T
print(kernel_gy)

plt.figure()
plt.imshow(img_gray)
plt.show()









