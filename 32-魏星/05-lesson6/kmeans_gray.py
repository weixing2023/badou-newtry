import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


'''
在OpenCV中，Kmeans()函数原型如下所示：
retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
    data表示聚类数据，最好是np.flloat32类型的N维点集
    K表示聚类类簇数
    bestLabels表示输出的整数数组，用于存储每个样本的聚类标签索引
    criteria表示迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
        其中，type有如下模式：
         —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
         —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
         —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
    attempts表示重复试验kmeans算法的次数，算法返回产生的最佳结果的标签
    flags表示初始中心的选择，两种方法是cv2.KMEANS_PP_CENTERS ;和cv2.KMEANS_RANDOM_CENTERS
    centers表示集群中心的输出矩阵，每个集群中心为一行数据
'''

img = cv2.imread("dog1.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用sklearn中KMeans方法
clf=KMeans(n_clusters=2)
kmeans_result = clf.fit_predict(img_gray.reshape(img_gray.shape[0]*img_gray.shape[1],1))
img_kmeans = kmeans_result.reshape(img_gray.shape[0], img_gray.shape[1])

# #停止条件 (type,max_iter,epsilon)
# criteria = (cv2.TERM_CRITERIA_EPS +
#             cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# #设置标签
# flags = cv2.KMEANS_RANDOM_CENTERS
#
# data = img_gray.reshape(img_gray.shape[0]*img_gray.shape[1],1)
# data = np.float32(data)
# #K-Means聚类 聚集成4类
# compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)
# print(labels)
#
# img_kmeans = labels.reshape(img_gray.shape[0], img_gray.shape[1])
# print(img_kmeans)

plt.figure()
plt.subplot(121)
plt.title("img_gray")
plt.imshow(img_gray)

plt.subplot(122)
plt.title("img_keans")
plt.imshow(img_kmeans)

plt.show()


