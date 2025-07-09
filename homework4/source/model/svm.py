"""
Copyright (C) 2025 傅祉珏 谢敬豪

:module: svm
:function: 存放SVM分类器
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


class SVM:
    """ 支持向量机(Support Vector Machine)分类器

    属性：
    learning_rate (float): 学习率。
    lr_decay (float): 学习率衰减系数。
    max_iter (int): 最大迭代次数。
    batch_size (int): 批量大小。
    seed (int): 随机种子。
    loss_fn (function): 损失函数。
    reg_strength (float): 正则化强度。
    weights (ndarray): 权重矩阵。
    bias (ndarray): 偏置项。
    scaler (StandardScaler): 数据标准化器。
    _X (ndarray): 训练数据特征。
    _y (ndarray): 训练数据标签。
    _X_test (ndarray): 测试数据特征。
    _y_test (ndarray): 测试数据标签。
    train_cm (ndarray): 训练集混淆矩阵。
    test_cm (ndarray): 测试集混淆矩阵。
    """

    def __init__(self, learning_rate=0.01, lr_decay=0.95, max_iter=100, batch_size=32,
                 random_seed=42, loss=None, reg_strength=0.001):
        """ 初始化SVM分类器

        参数：
        learning_rate (float): 学习率，默认为0.01。
        lr_decay (float): 学习率衰减系数，默认为0.95。
        max_iter (int): 最大迭代次数，默认为100。
        batch_size (int): 批量大小，默认为32。
        random_seed (int): 随机种子，默认为42。
        loss (function): 损失函数，默认为None。
        reg_strength (float): L2正则化强度，默认为0.001。
        """
        self.learning_rate = learning_rate  # 学习率
        self.lr_decay = lr_decay  # 学习率衰减
        self.max_iter = max_iter  # 最大迭代次数
        self.batch_size = batch_size  # 批量大小
        self.seed = random_seed  # 随机种子
        self.loss_fn = loss  # 损失函数
        self.reg_strength = reg_strength  # 正则化强度

        self.weights = None  # 权重矩阵
        self.bias = None  # 偏置项
        self.scaler = StandardScaler()  # 数据标准化器
        self._X = None  # 训练数据特征
        self._y = None  # 训练数据标签
        self._X_test = None  # 测试数据特征
        self._y_test = None  # 测试数据标签

        self.train_cm = None  # 训练集混淆矩阵
        self.test_cm = None  # 测试集混淆矩阵

    def _predict_scores(self, X):
        """ 计算分类得分

        参数：
        X (ndarray): 输入特征矩阵。

        返回：
        ndarray: 分类得分矩阵。
        """
        return np.dot(X, self.weights) + self.bias

    def _predict(self, X):
        """ 预测类别标签

        参数：
        X (ndarray): 输入特征矩阵。

        返回：
        ndarray: 预测的类别标签。
        """
        scores = self._predict_scores(X)
        return np.argmax(scores, axis=1)

    def train(self, X=None, y=None):
        """ 训练SVM模型

        参数：
        X (ndarray): 训练数据特征。
        y (ndarray): 训练数据标签。

        返回：
        tuple: (训练准确率, 训练损失)

        异常：
        ValueError: 当未提供训练数据时抛出。
        """
        if X is not None and y is not None:
            self._X = X
            self._y = y

        if self._X is None or self._y is None:
            raise ValueError("Must provide training data to train().")

        # 数据预处理
        np.random.seed(self.seed)
        X, y = shuffle(self._X, self._y, random_state=self.seed)
        X = self.scaler.fit_transform(X)

        n_samples, n_features = X.shape
        n_classes = np.max(y) + 1

        # 初始化模型参数
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)

        # 训练循环
        for epoch in range(self.max_iter):
            X, y = shuffle(X, y)  # 每个epoch打乱数据

            # 小批量训练
            for i in range(0, n_samples, self.batch_size):
                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]

                # 计算分类得分
                scores = np.dot(X_batch, self.weights) + self.bias

                # 计算正确类别的得分
                correct_scores = scores[np.arange(len(y_batch)), y_batch].reshape(-1, 1)

                # 计算间隔(margin)
                margins = scores - correct_scores + 1  # delta=1
                margins = np.maximum(0, margins)  # ReLU
                margins[np.arange(len(y_batch)), y_batch] = 0  # 忽略正确类别

                # 计算梯度
                mask = (margins > 0).astype(float)
                row_sum = np.sum(mask, axis=1)
                mask[np.arange(len(y_batch)), y_batch] = -row_sum

                # 权重梯度(含L2正则化)
                dW = np.dot(X_batch.T, mask) / len(y_batch) + self.reg_strength * self.weights
                db = np.sum(mask, axis=0) / len(y_batch)

                # 参数更新
                self.weights -= self.learning_rate * dW
                self.bias -= self.learning_rate * db

            # 学习率衰减
            self.learning_rate *= self.lr_decay

        # 计算训练指标
        train_scores = self._predict_scores(X)
        train_loss = self.loss_fn(y, train_scores)
        train_acc = np.mean(np.argmax(train_scores, axis=1) == y)
        self.train_cm = confusion_matrix(y, np.argmax(train_scores, axis=1))

        return train_acc, train_loss

    def evaluate(self, X=None, y=None):
        """ 评估模型性能

        参数：
        X (ndarray): 测试数据特征。
        y (ndarray): 测试数据标签。

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
        scores = self._predict_scores(X)

        # 计算评估指标
        loss = self.loss_fn(self._y_test, scores)
        acc = np.mean(np.argmax(scores, axis=1) == self._y_test)
        self.test_cm = confusion_matrix(self._y_test, np.argmax(scores, axis=1))

        return acc, loss

    def predict(self, X):
        """ 对新样本进行预测

        参数：
        X (ndarray): 待预测数据特征。

        返回：
        ndarray: 预测的类别标签。
        """
        X = self.scaler.transform(X)
        X = np.array(X, dtype=np.float64)
        return self._predict(X)

    def predict_proba(self, X):
        """ 预测样本属于各个类别的概率

        参数：
        X (ndarray): 待预测数据特征。

        返回：
        ndarray: 类别概率矩阵。
        """
        X = self.scaler.transform(X)
        X = np.array(X, dtype=np.float64)
        scores = self._predict_scores(X)
        # 数值稳定性处理
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        proba = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return proba

    def confusion_matrix(self):
        """ 获取混淆矩阵

        返回：
        tuple: (训练集混淆矩阵, 测试集混淆矩阵)
        """
        return self.train_cm, self.test_cm
