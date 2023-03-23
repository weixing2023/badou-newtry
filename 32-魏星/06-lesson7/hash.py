import numpy as np
import cv2


'''
均值hash，步骤：
1、缩放
2、灰度化
3、求平均
4、比较
5、生成hash
'''
def aHash(img):
    #缩放
    img_resize = cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
    #灰度化
    img_gray = cv2.cvtColor(img_resize,cv2.COLOR_BGR2GRAY)
    #求平均
    mean_num = np.mean(img_gray)
    #比较、生成hash
    hash_str = ''
    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            if img_gray[i,j] > mean_num:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'

    return hash_str


'''
差值hash，步骤：
1、缩放
2、灰度化
3、比较：行内比较，每行内，前一个像素与后一个像素值比较；行间不比较
4、生成hash
'''
def bHash(img):
    #缩放
    img_resize = cv2.resize(img,(8,9),interpolation=cv2.INTER_CUBIC)
    #灰度化
    img_gray = cv2.cvtColor(img_resize,cv2.COLOR_BGR2GRAY)
    #比较、生成hash
    hash_str = ''
    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]-1):
            if img_gray[i,j] > img_gray[i,j+1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'

    return hash_str


def cmp_hash(hash0,hash1):
    s = 0
    for i in range(hash0.__len__()):
        if hash0[i] == hash1[i]:
            s += 1
    return s, s/hash0.__len__()


# 汉明距离计算相似度
img1 = cv2.imread("dog1.jpg")
img2 = cv2.imread("dog2.jpg")
hash0 = aHash(img1)
hash1 = aHash(img2)
print(hash0)
print(hash1)
print("均值哈希，汉明距离和相似度为",cmp_hash(hash0,hash1))

img1 = cv2.imread("dog1.jpg")
img2 = cv2.imread("dog2.jpg")
hash0 = bHash(img1)
hash1 = bHash(img2)
print(hash0)
print(hash1)
print("差值哈希，汉明距离和相似度为",cmp_hash(hash0,hash1))



