import matplotlib.pyplot as plt
import random
import cv2

'''
高斯噪声

Pout = Pin + random.gauss
'''

def gaussian_nosie(img,means,sigma,snr):
    w,h = img.shape
    img_result = img

    noiseNum = int(snr*w*h)
    for i in range(noiseNum):
        randX = random.randint(0, w-1)
        randY = random.randint(0, h-1)
        img_result[randX,randY] = img[randX,randY] + random.gauss(means,sigma)

        if img_result[randX,randY] < 0 :
            img_result[randX,randY] = 0
        elif img_result[randX,randY] > 255:
            img_result[randX, randY] = 255

    return img_result


img = cv2.imread("dog2.jpg")
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_result = gaussian_nosie(img_gray,0.6,1.2,0.8)
img_gray1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #img_gray已经被修改

plt.figure()
plt.subplot(121)
plt.title("normal figure")
plt.imshow(img_gray1)

plt.subplot(122)
plt.title("gaussian noise figure")
plt.imshow(img_result)

plt.show()


















