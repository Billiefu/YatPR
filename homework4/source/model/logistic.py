"""
Copyright (C) 2025 傅祉珏 杨程骏

:module: logistic
:function: 存放逻辑斯蒂分类器
:author: 傅祉珏，杨程骏
:date: 2025-07-03
:license: AGPLv3 + 附加限制条款（禁止商用）

本代码依据 GNU Affero 通用公共许可证 v3 (AGPLv3) 授权，附加条款见项目根目录 ADDITIONAL_TERMS.md。
- 禁止商业用途（包括但不限于销售、集成至商业产品）
- 学术使用需在代码注释或文档中明确标注来源

AGPLv3 完整文本见 LICENSE 文件或 <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


class LogisticRegression:
    """ 多分类逻辑回归分类器

    属性：
    scaler (StandardScaler): 数据标准化器。
    learning_rate (float): 学习率。
    lr_decay (float): 学习率衰减系数。
    max_iter (int): 最大迭代次数。
    batch_size (int): 批量大小。
    loss_fn (function): 损失函数。
    seed (int): 随机种子。
    weights (ndarray): 模型权重矩阵。
    bias (ndarray): 模型偏置项。
    train_cm (ndarray): 训练集混淆矩阵。
    test_cm (ndarray): 测试集混淆矩阵。
    """

    def __init__(self, learning_rate=0.1, lr_decay=0.95, max_iter=100, batch_size=32,
                 random_seed=42, loss=None):
        """ 初始化逻辑回归分类器

        参数：
        learning_rate (float): 学习率，默认为0.1。
        lr_decay (float): 学习率衰减系数，默认为0.95。
        max_iter (int): 最大迭代次数，默认为100。
        batch_size (int): 批量大小，默认为32。
        random_seed (int): 随机种子，默认为42。
        loss (function): 损失函数，默认为None。
        """
        self.scaler = StandardScaler()  # 数据标准化器
        self.learning_rate = learning_rate  # 学习率
        self.lr_decay = lr_decay  # 学习率衰减系数
        self.max_iter = max_iter  # 最大迭代次数
        self.batch_size = batch_size  # 批量大小
        self.loss_fn = loss  # 损失函数
        self.seed = random_seed  # 随机种子
        self.weights = None  # 权重矩阵
        self.bias = None  # 偏置项

        self.train_cm = None  # 训练集混淆矩阵
        self.test_cm = None  # 测试集混淆矩阵

    def _softmax(self, logits):
        """ 计算softmax概率

        参数：
        logits (ndarray): 未归一化的预测值。

        返回：
        ndarray: 归一化的概率分布。
        """
        logits = np.array(logits, dtype=np.float64)
        # 数值稳定性的处理：减去最大值
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)  # 归一化

    def _predict_proba(self, X):
        """ 预测样本属于各个类别的概率

        参数：
        X (ndarray): 样本特征矩阵。

        返回：
        ndarray: 概率矩阵，shape=(n_samples, n_classes)。
        """
        logits = np.dot(X, self.weights) + self.bias  # 线性变换
        return self._softmax(logits)  # softmax归一化

    def _predict(self, X):
        """ 预测样本类别

        参数：
        X (ndarray): 样本特征矩阵。

        返回：
        ndarray: 预测的类别标签。
        """
        return np.argmax(self._predict_proba(X), axis=1)  # 取概率最大的类别

    def train(self, X=None, y=None):
        """ 训练逻辑回归模型

        参数：
        X (ndarray): 训练数据特征。
        y (ndarray): 训练数据标签。

        返回：
        tuple: (训练准确率, 训练损失)

        异常：
        ValueError: 当未提供训练数据时抛出。
        """
        if X is None or y is None:
            raise ValueError("You must pass X and y to train()")

        # 数据预处理
        np.random.seed(self.seed)
        X, y = shuffle(X, y, random_state=self.seed)
        X = self.scaler.fit_transform(X)

        # 初始化模型参数
        n_samples, n_features = X.shape
        n_classes = np.max(y) + 1
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)

        history_loss = []  # 记录损失历史

        # 训练循环
        for epoch in range(self.max_iter):
            X, y = shuffle(X, y)  # 每个epoch打乱数据

            # 小批量梯度下降
            for i in range(0, n_samples, self.batch_size):
                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]

                # 前向传播
                logits = np.dot(X_batch, self.weights) + self.bias
                probs = self._softmax(logits)

                # 将标签转换为one-hot编码
                y_one_hot = np.zeros_like(probs)
                y_one_hot[np.arange(len(y_batch)), y_batch] = 1

                # 计算梯度
                grad_logits = probs - y_one_hot
                grad_w = np.dot(X_batch.T, grad_logits) / len(y_batch)
                grad_b = np.sum(grad_logits, axis=0) / len(y_batch)

                # 参数更新
                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b

            # 学习率衰减
            self.learning_rate *= self.lr_decay

            # 记录损失
            y_pred_proba = self._predict_proba(X)
            loss = self.loss_fn(y, y_pred_proba)
            history_loss.append(loss)

        # 计算最终训练指标
        train_acc = np.mean(self._predict(X) == y)
        train_loss = history_loss[-1]
        self.train_cm = confusion_matrix(y, self._predict(X))

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
        if X is None or y is None:
            raise ValueError("You must pass X and y to evaluate()")

        # 数据标准化
        X = self.scaler.transform(X)

        # 预测和评估
        y_pred = self._predict(X)
        y_pred_proba = self._predict_proba(X)

        acc = np.mean(y_pred == y)
        loss = self.loss_fn(y, y_pred_proba)
        self.test_cm = confusion_matrix(y, y_pred)

        return acc, loss

    def predict(self, X):
        """ 对新样本进行预测

        参数：
        X (ndarray): 待预测数据特征。

        返回：
        ndarray: 预测的类别标签数组。
        """
        X = self.scaler.transform(X)
        X = np.asarray(X, dtype=np.float64)
        return self._predict(X)

    def predict_proba(self, X):
        """ 预测样本属于各个类别的概率

        参数：
        X (ndarray): 待预测数据特征。

        返回：
        ndarray: 概率矩阵，shape=(n_samples, n_classes)。
        """
        X = self.scaler.transform(X)
        X = np.asarray(X, dtype=np.float64)
        return self._predict_proba(X)

    def confusion_matrix(self):
        """ 获取混淆矩阵

        返回：
        tuple: (训练集混淆矩阵, 测试集混淆矩阵)
        """
        return self.train_cm, self.test_cm
