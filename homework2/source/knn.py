"""
@module knn
@function 存放KNN分类器
@author 傅祉珏
@date 2025年4月9日
"""

import numpy as np


class KNN:
    """ K-最近邻（KNN）分类器类
    该类实现了KNN分类算法，包括训练、预测和评估方法。

    属性：
    k (int): 设定的邻居数量，默认为3。
    train_data (ndarray): 训练数据集的特征矩阵。
    train_label (ndarray): 训练数据集的标签向量。
    """

    def __init__(self, k=3):
        """ 初始化KNN分类器

        参数：
        k (int): 设定的邻居数量，默认为3。
        """
        self.k = k  # 设置邻居数量
        self.train_data = None  # 初始化训练数据
        self.train_label = None  # 初始化训练标签

    def fit(self, X, y):
        """ 训练KNN分类器

        参数：
        X (ndarray): 训练数据特征矩阵，每行是一个样本，每列是一个特征。
        y (ndarray): 训练数据的标签向量，每个元素是一个类别标签。
        """
        self.train_data = X  # 存储训练数据
        self.train_label = y  # 存储训练标签

    def predict(self, X):
        """ 对输入数据进行预测

        参数：
        X (ndarray): 待预测的数据特征矩阵，每行是一个样本，每列是一个特征。

        返回：
        ndarray: 每个输入样本的预测类别标签。
        """
        preds = []  # 用于存储每个样本的预测结果
        for x in X:
            # 计算当前样本与所有训练样本的欧氏距离
            distances = np.linalg.norm(self.train_data - x, axis=1)

            # 获取与当前样本距离最近的 k 个训练样本的索引
            k_indices = np.argsort(distances)[:self.k]

            # 获取这 k 个训练样本的标签
            k_labels = self.train_label[k_indices]

            # 投票：选择出现频率最高的类别作为预测结果
            pred = np.bincount(k_labels).argmax()  # 找到出现次数最多的标签
            preds.append(pred)  # 将预测结果添加到结果列表中

        return np.array(preds)  # 返回所有样本的预测结果

    def score(self, X, y):
        """ 计算分类准确率

        参数：
        X (ndarray): 测试数据特征矩阵，每行是一个样本，每列是一个特征。
        y (ndarray): 测试数据的真实标签向量，每个元素是一个类别标签。

        返回：
        float: 分类准确率，即预测标签与真实标签一致的比例。
        """
        y_pred = self.predict(X)  # 获取预测结果
        return np.mean(y_pred == y)  # 计算预测准确率
