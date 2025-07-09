"""
Copyright (C) 2025 傅祉珏 谢敬豪

:module: knn
:function: 存放KNN分类器
:author: 傅祉珏，谢敬豪
:date: 2025-07-03
:license: AGPLv3 + 附加限制条款（禁止商用）

本代码依据 GNU Affero 通用公共许可证 v3 (AGPLv3) 授权，附加条款见项目根目录 ADDITIONAL_TERMS.md。
- 禁止商业用途（包括但不限于销售、集成至商业产品）
- 学术使用需在代码注释或文档中明确标注来源

AGPLv3 完整文本见 LICENSE 文件或 <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle


class KNN:
    """ K近邻(K-Nearest Neighbors)分类器实现

    属性：
    num_classes (int): 类别数量。
    learning_rate (float): 学习率(保留参数，实际不使用)。
    lr_decay (float): 学习率衰减系数(保留参数)。
    max_iter (int): 最大迭代次数(保留参数)。
    batch_size (int): 批量大小(保留参数)。
    random_seed (int): 随机种子。
    loss_fn (function): 损失函数。
    k (int): 近邻数。
    distance_metric (str): 距离度量方法。
    scaler (StandardScaler): 数据标准化器。
    X_train (ndarray): 训练数据特征。
    y_train (ndarray): 训练数据标签。
    X_test (ndarray): 测试数据特征。
    y_test (ndarray): 测试数据标签。
    train_cm (ndarray): 训练集混淆矩阵。
    test_cm (ndarray): 测试集混淆矩阵。
    """

    def __init__(self, learning_rate=0.01, lr_decay=0.9, max_iter=1, batch_size=32,
                 random_seed=42, loss=None, k=5, distance_metric='euclidean'):
        """ 初始化KNN分类器

        参数：
        learning_rate (float): 学习率(保留参数)，默认为0.01。
        lr_decay (float): 学习率衰减系数(保留参数)，默认为0.9。
        max_iter (int): 最大迭代次数(保留参数)，默认为1。
        batch_size (int): 批量大小(保留参数)，默认为32。
        random_seed (int): 随机种子，默认为42。
        loss (function): 损失函数，默认为None。
        k (int): 近邻数，默认为5。
        distance_metric (str): 距离度量方法，默认为'euclidean'。
        """
        self.num_classes = None  # 类别数量
        self.learning_rate = learning_rate  # 无用，仅接口保留
        self.lr_decay = lr_decay  # 学习率衰减
        self.max_iter = max_iter  # 最大迭代次数
        self.batch_size = batch_size  # 批量大小
        self.random_seed = random_seed  # 随机种子

        self.loss_fn = loss  # 损失函数
        self.k = k  # 近邻数
        self.distance_metric = distance_metric  # 距离度量方法

        self.scaler = StandardScaler()  # 数据标准化器
        self.X_train = None  # 训练数据特征
        self.y_train = None  # 训练数据标签
        self.X_test = None  # 测试数据特征
        self.y_test = None  # 测试数据标签

        self.train_cm = None  # 训练集混淆矩阵
        self.test_cm = None  # 测试集混淆矩阵

    def _distance(self, x1, x2):
        """ 计算两个样本之间的距离

        参数：
        x1 (ndarray): 第一个样本或样本集。
        x2 (ndarray): 第二个样本。

        返回：
        ndarray: 样本之间的距离。

        异常：
        ValueError: 当使用不支持的距离度量方法时抛出。
        """
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))  # 欧氏距离
        else:
            raise ValueError("Unsupported distance metric")

    def _predict_one(self, x):
        """ 预测单个样本的类别

        参数：
        x (ndarray): 单个样本特征。

        返回：
        int: 预测的类别标签。
        """
        # 计算与所有训练样本的距离
        dists = self._distance(self.X_train, x)
        # 获取k个最近邻的索引
        idx = np.argsort(dists)[:self.k]
        # 获取最近邻的标签
        neighbors = self.y_train[idx]
        # 统计各类别出现次数
        counts = np.bincount(neighbors)
        # 返回出现次数最多的类别
        return np.argmax(counts)

    def _predict(self, X):
        """ 预测多个样本的类别

        参数：
        X (ndarray): 样本特征矩阵。

        返回：
        ndarray: 预测的类别标签数组。
        """
        return np.array([self._predict_one(x) for x in X])

    def train(self, X=None, y=None):
        """ 训练KNN模型(实际仅存储数据)

        参数：
        X (ndarray): 训练数据特征，默认为None。
        y (ndarray): 训练数据标签，默认为None。

        返回：
        tuple: (训练准确率, 训练损失)

        异常：
        ValueError: 当未提供训练数据时抛出。
        """
        if X is not None and y is not None:
            self.X_train = X
            self.y_train = y
            self.y_train = np.array(y)
            self.num_classes = len(np.unique(y))
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data must be provided.")

        # 数据预处理
        np.random.seed(self.random_seed)
        self.X_train, self.y_train = shuffle(self.X_train, self.y_train, random_state=self.random_seed)
        self.X_train = self.scaler.fit_transform(self.X_train)

        # 评估训练数据
        y_pred = self._predict(self.X_train)
        acc = np.mean(y_pred == self.y_train)
        loss = self.loss_fn(self.y_train, y_pred)

        self.train_cm = confusion_matrix(self.y_train, y_pred)

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
            self.X_test = X
            self.y_test = y
            self.y_test = np.array(y)
        if self.X_test is None or self.y_test is None:
            raise ValueError("Test data must be provided.")

        # 数据标准化和预测
        X_scaled = self.scaler.transform(self.X_test)
        y_pred = self._predict(X_scaled)

        # 计算评估指标
        acc = np.mean(y_pred == self.y_test)
        loss = self.loss_fn(self.y_test, y_pred)
        self.test_cm = confusion_matrix(self.y_test, y_pred)

        return acc, loss

    def predict(self, X):
        """ 对新样本进行预测

        参数：
        X (ndarray): 待预测数据特征。

        返回：
        ndarray: 预测的类别标签数组。
        """
        X = self.scaler.transform(X)
        X = np.array(X, dtype=np.float64)
        return self._predict(X)

    def predict_proba(self, X):
        """ 预测样本属于各个类别的概率(硬分配)

        参数：
        X (ndarray): 待预测数据特征。

        返回：
        ndarray: 概率矩阵，shape=(n_samples, n_classes)。
        """
        X = self.scaler.transform(X)
        X = np.asarray(X, dtype=np.float64)
        y_pred = self._predict(X)
        proba = np.zeros((len(y_pred), self.num_classes))
        proba[np.arange(len(y_pred)), y_pred] = 1  # 硬分配概率(0或1)
        return proba

    def confusion_matrix(self):
        """ 获取混淆矩阵

        返回：
        tuple: (训练集混淆矩阵, 测试集混淆矩阵)
        """
        return self.train_cm, self.test_cm
