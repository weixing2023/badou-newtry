import numpy as np

class PCA():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self,X):
        self.n_features = X.shape[1] #样本矩阵维度
        # 求样本矩阵的协方差矩阵(按列中心化后并按维度计算协方差)
        X = X - X.mean(axis=0)
        self.covariance = np.dot(X.T,X)/(X.shape[0]-1)
        # 求协方差矩阵的特征值和特征向量
        eig_vals, eig_vectors = np.linalg.eig(self.covariance)
        # 获得降序排列特征值的序号
        idx = np.argsort(-eig_vals)
        # 降维矩阵
        self.components = eig_vectors[:, idx[:self.n_components]]
        variance_ratio = np.sum([eig_vals[idx[i]] for i in range(self.n_components)]) / np.sum(eig_vals)
        variance_ratios = np.array([eig_vals[idx[i]] for i in range(self.n_components)]/np.sum(eig_vals))
        print("前%d个特征值的贡献率分别为%s" % (self.n_components, variance_ratios))
        # 对X进行降维
        return np.dot(X, self.components),variance_ratio

pca = PCA(n_components=3)
X = np.array([[-1,2,66,-1],[-2,6,58,-1],[-3,8,45,-2],[1,9,36,1],[2,10,62,1],[3,5,83,2]])
newX,variance_ratio = pca.fit_transform(X)
# 输出降维后的数据
print(newX)
print("前K个特征值的贡献率为\n",variance_ratio)