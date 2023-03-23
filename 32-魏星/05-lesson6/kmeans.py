import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 生成数据集
np.random.seed(20)
X1 = np.random.random(20)
X2 = np.random.random(20)
X = np.array((X1,X2)).T

clf = KMeans(n_clusters=4)
kmeans_result = clf.fit_predict(X)
# print(kmeans_result)

plt.scatter(X1, X2, c=kmeans_result, marker='*')
plt.legend(["a","b","c","d"])
plt.show()