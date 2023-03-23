import numpy as np

'''
实现pca，
X为样本矩阵，列表示维度，行表示样本数
K阶降维矩阵的K值，即保留K个特征值
'''
class PCA(object):
    def __init__(self,X,K):
        self.X = X          #样本矩阵
        self.centrX = []    #样本矩阵的中心化矩阵
        self.K = K          #K阶降维矩阵的K值
        self.C = []         #中心化样本矩阵的协方差矩阵
        self.W = []         #K阶降维矩阵
        self.XW = []        #样本矩阵的K阶降维矩阵

        self.centrX = self._centralized()
        self.C = self._cov()
        self.W,self.variance_ratio = self._W()
        self.XW = self._XW()

    # 矩阵的中心化
    def _centralized(self):
        print('样本矩阵X：\n', self.X)
        centrX = []
        # 按维度(列)求样本矩阵的均值
        # mean = np.array([np.mean(attr) for attr in self.X.T])
        mean = np.mean(self.X,0)
        print('样本集的特征均值：\n', mean)
        centrX = self.X - mean  ##样本集的中心化
        print('样本矩阵X的中心化centrX:\n', centrX)
        return centrX

    #样本矩阵的协方差矩阵
    def _cov(self):
        # 样本集的样例总数
        ns = np.shape(self.centrX)[0]
        # 样本矩阵的协方差矩阵C
        C = np.dot(self.centrX.T, self.centrX) / (ns - 1)
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C

    #样本矩阵的K阶降维矩阵W，shape=(n,k),n为维度，k为降维矩阵的特征维度
    def _W(self):
        # 先求X的协方差矩阵C的特征值和特征向量
        a, b = np.linalg.eig(self.C)
        print('样本集的协方差矩阵C的特征值：\n', a)
        print('样本集的协方差矩阵C的特征向量：\n', b)
        # 给出特征值降序的topK的索引序列，返回的是特征值向量排序后的索引
        ind = np.argsort(-1 * a)
        # 构建K阶降维的降维转换矩阵W
        WT = [b[:, ind[i]] for i in range(self.K)] #按列获取然后转置
        W = np.transpose(WT)
        # W = b[:, ind[:self.K]]
        #计算前K个特征值的贡献率
        variance_ratio = np.sum([a[ind[i]] for i in range(self.K)])/np.sum(a)
        variance_ratios = np.array([a[ind[i]] for i in range(self.K)] / np.sum(a))
        print("前%d个特征值的贡献率分别为%s" % (self.K, variance_ratios))
        print('%d阶降维转换矩阵W：\n' % self.K, W)
        return W,variance_ratio

    #按照XW=X*W求降维矩阵XW，shape=(m,k),m是样本总数，k是降维矩阵中特征维度总数
    def _XW(self):
        XW = np.dot(self.X, self.W)
        print('X shape:', np.shape(self.X))
        print('W shape:', np.shape(self.W))
        print('XW shape:', np.shape(XW))
        print('前K个特征值贡献率为：',self.variance_ratio)
        print('样本矩阵X的降维矩阵XW:\n', XW)
        return XW


if __name__ == '__main__':
    '10个样本3个特征的样本集，行为样例，列为特征维度'
    X = np.array([
        [10, 15, 29],
        [15, 46, 13],
        [23, 21, 30],
        [11, 9, 35],
        [42, 45, 11],
        [9, 48, 5],
        [11, 21, 14],
        [8, 5, 15],
        [11, 12, 21],
        [21, 20, 25]])
    K = np.shape(X)[1] - 1
    print('样本集：\n', X)
    pca = PCA(X, K)


