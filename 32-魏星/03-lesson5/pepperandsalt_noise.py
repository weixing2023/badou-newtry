import matplotlib.pyplot as plt
import random
import cv2

'''
椒盐噪声
'''

def pepperandsalt_nosie(img,snr):
    w,h = img.shape
    img_result = img.copy() #深copy，不改变img

    noiseNum = int(snr*w*h)
    for i in range(noiseNum):
        randX = random.randint(0, w-1)
        randY = random.randint(0, h-1)

        if int(w*0.2) < randX < int(w*0.8) and int(h*0.2) < randY < int(h*0.8):
            if random.random() <= 0.5:
                img_result[randX, randY] = 0
            else:
                img_result[randX, randY] = 255

    return img_result


img = cv2.imread("dog1.jpg")
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_result = pepperandsalt_nosie(img_gray,0.2)

plt.figure()
plt.subplot(121)
plt.title("normal figure")
plt.imshow(img_gray)

plt.subplot(122)
plt.title("pepperAndSalt noise figure")
plt.imshow(img_result)

plt.show()

