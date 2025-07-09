"""
Copyright (C) 2025 傅祉珏

:module: kmeans
:function: 存放K-means分类器
:author: 傅祉珏
:date: 2025-07-03
:license: AGPLv3 + 附加限制条款（禁止商用）

本代码依据 GNU Affero 通用公共许可证 v3 (AGPLv3) 授权，附加条款见项目根目录 ADDITIONAL_TERMS.md。
- 禁止商业用途（包括但不限于销售、集成至商业产品）
- 学术使用需在代码注释或文档中明确标注来源

AGPLv3 完整文本见 LICENSE 文件或 <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


class Kmeans:
    """ K-means聚类算法实现

    属性：
    num_clusters (int): 聚类簇数。
    learning_rate (float): 学习率(保留参数，实际不使用)。
    lr_decay (float): 学习率衰减系数(保留参数)。
    max_iter (int): 最大迭代次数。
    batch_size (int): 批量大小(保留参数)。
    random_seed (int): 随机种子。
    loss_fn (function): 损失函数。
    tol (float): 收敛阈值。
    centroids (ndarray): 聚类中心点。
    _X (ndarray): 训练数据特征。
    _y (ndarray): 训练数据标签。
    _X_test (ndarray): 测试数据特征。
    _y_test (ndarray): 测试数据标签。
    scaler (StandardScaler): 数据标准化器。
    train_cm (ndarray): 训练集混淆矩阵。
    test_cm (ndarray): 测试集混淆矩阵。
    """

    def __init__(self, learning_rate=0.01, lr_decay=0.95, max_iter=1000, batch_size=32,
                 random_seed=42, loss=None, tol=1e-4):
        """ 初始化K-means分类器

        参数：
        learning_rate (float): 学习率(保留参数)，默认为0.01。
        lr_decay (float): 学习率衰减系数(保留参数)，默认为0.95。
        max_iter (int): 最大迭代次数，默认为1000。
        batch_size (int): 批量大小(保留参数)，默认为32。
        random_seed (int): 随机种子，默认为42。
        loss (function): 损失函数，默认为None。
        tol (float): 收敛阈值，默认为1e-4。
        """
        self.num_clusters = None  # 聚类簇数
        self.learning_rate = learning_rate  # 接口统一，实际不使用
        self.lr_decay = lr_decay  # 学习率衰减
        self.max_iter = max_iter  # 最大迭代次数
        self.batch_size = batch_size  # 批量大小
        self.random_seed = random_seed  # 随机种子
        self.loss_fn = loss  # 损失函数
        self.tol = tol  # 收敛阈值

        self.centroids = None  # 聚类中心
        self._X = None  # 训练数据特征
        self._y = None  # 训练数据标签
        self._X_test = None  # 测试数据特征
        self._y_test = None  # 测试数据标签
        self.scaler = StandardScaler()  # 数据标准化器

        self.train_cm = None  # 训练集混淆矩阵
        self.test_cm = None  # 测试集混淆矩阵

    def _init_centroids(self, X):
        """ 初始化聚类中心

        参数：
        X (ndarray): 样本特征矩阵。

        返回：
        ndarray: 随机选择的初始聚类中心。
        """
        np.random.seed(self.random_seed)
        indices = np.random.choice(len(X), self.num_clusters, replace=False)
        return X[indices]

    def _assign_clusters(self, X):
        """ 分配样本到最近的聚类中心

        参数：
        X (ndarray): 样本特征矩阵。

        返回：
        ndarray: 每个样本所属的簇标签。
        """
        X = np.asarray(X, dtype=np.float64)
        # 计算每个样本到所有聚类中心的距离
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], axis=2)
        return np.argmin(distances, axis=1)  # 返回最近簇的索引

    def _update_centroids(self, X, labels):
        """ 更新聚类中心

        参数：
        X (ndarray): 样本特征矩阵。
        labels (ndarray): 样本所属簇标签。

        返回：
        ndarray: 更新后的聚类中心。
        """
        new_centroids = np.zeros_like(self.centroids)
        for k in range(self.num_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                new_centroids[k] = cluster_points.mean(axis=0)  # 计算簇内均值
            else:
                new_centroids[k] = self.centroids[k]  # 避免空簇
        return new_centroids

    def _match_clusters(self, y_true, y_pred):
        """ 使用匈牙利算法匹配聚类标签与真实标签

        参数：
        y_true (ndarray): 真实标签。
        y_pred (ndarray): 预测标签。

        返回：
        ndarray: 重新映射后的预测标签。
        """
        cm = confusion_matrix(y_true, y_pred)
        # 匈牙利算法求最大匹配(传入负值找最大值)
        row_ind, col_ind = linear_sum_assignment(-cm)
        mapping = dict(zip(col_ind, row_ind))  # 创建映射字典

        # 将聚类标签映射为真实标签对应的标签
        new_preds = np.vectorize(lambda x: mapping.get(x, x))(y_pred)
        return new_preds

    def _predict(self, X):
        """ 内部预测方法

        参数：
        X (ndarray): 样本特征矩阵。

        返回：
        ndarray: 预测的簇标签。
        """
        return self._assign_clusters(X)

    def train(self, X=None, y=None):
        """ 训练K-means模型

        参数：
        X (ndarray): 训练数据特征，默认为None。
        y (ndarray): 训练数据标签，默认为None。

        返回：
        tuple: (训练准确率, 训练损失)

        异常：
        ValueError: 当未提供训练数据时抛出。
        """
        if X is not None and y is not None:
            self._X = X
            self._y = y
            self.num_clusters = len(np.unique(y))

        if self._X is None:
            raise ValueError("Must provide training data to train().")

        # 数据标准化
        X = self.scaler.fit_transform(self._X)
        y = self._y if self._y is not None else None

        # 初始化聚类中心
        self.centroids = self._init_centroids(X)

        # K-means迭代过程
        for i in range(self.max_iter):
            labels = self._assign_clusters(X)  # 分配簇标签
            new_centroids = self._update_centroids(X, labels)  # 更新聚类中心
            shift = np.linalg.norm(self.centroids - new_centroids)  # 计算中心点移动距离
            self.centroids = new_centroids

            # 检查收敛
            if shift < self.tol:
                break

        # 如果有标签数据，计算评估指标
        if y is not None:
            pred_labels = self._predict(X)
            mapped_labels = self._match_clusters(y, pred_labels)  # 标签匹配
            acc = np.mean(mapped_labels == y)
            loss = self.loss_fn(y, mapped_labels)

            self.train_cm = confusion_matrix(y, mapped_labels)
        else:
            acc = 0.0
            loss = 0.0

        return acc, loss

    def evaluate(self, X=None, y=None):
        """ 评估模型性能

        参数：
        X (ndarray): 测试数据特征，默认为None。
        y (ndarray): 测试数据标签，默认为None。

        返回：
        tuple: (测试准确率, 测试损失)

        异常：
        ValueError: 当未提供测试数据时抛出。
        """
        if X is not None and y is not None:
            self._X_test = X
            self._y_test = y

        if self._X_test is None or self._y_test is None:
            raise ValueError("Must provide test data to evaluate().")

        # 数据标准化和预测
        X = self.scaler.transform(self._X_test)
        y = self._y_test

        pred_labels = self._predict(X)
        mapped_labels = self._match_clusters(y, pred_labels)  # 标签匹配

        # 计算评估指标
        acc = np.mean(mapped_labels == y)
        loss = self.loss_fn(y, mapped_labels)
        self.test_cm = confusion_matrix(y, mapped_labels)

        return acc, loss

    def predict(self, X):
        """ 对新样本进行预测

        参数：
        X (ndarray): 待预测数据特征。

        返回：
        ndarray: 预测的簇标签数组。
        """
        X = self.scaler.transform(X)
        X = np.asarray(X, dtype=np.float64)
        return self._predict(X)

    def predict_proba(self, X):
        """ 预测样本属于各个簇的概率(硬分配)

        参数：
        X (ndarray): 待预测数据特征。

        返回：
        ndarray: 概率矩阵，shape=(n_samples, n_clusters)。
        """
        X = self.scaler.transform(X)
        X = np.asarray(X, dtype=np.float64)
        y_pred = self._predict(X)
        proba = np.zeros((len(y_pred), self.num_clusters))
        proba[np.arange(len(y_pred)), y_pred] = 1  # 硬分配概率(0或1)
        return proba

    def confusion_matrix(self):
        """ 获取混淆矩阵

        返回：
        tuple: (训练集混淆矩阵, 测试集混淆矩阵)
        """
        return self.train_cm, self.test_cm
