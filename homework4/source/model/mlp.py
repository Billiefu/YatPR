"""
Copyright (C) 2025 傅祉珏

:module: mlp
:function: 存放多层感知机分类器
:author: 傅祉珏
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


class MLP:
    """ 多层感知机(Multi-Layer Perceptron)分类器

    属性：
    learning_rate (float): 学习率。
    lr_decay (float): 学习率衰减系数。
    max_iter (int): 最大训练迭代次数。
    batch_size (int): 批量大小。
    seed (int): 随机种子。
    loss_fn (function): 损失函数。
    hidden_size (int): 隐藏层神经元数量。
    activation_type (str): 激活函数类型('relu'或'tanh')。
    _X (ndarray): 训练数据特征。
    _y (ndarray): 训练数据标签。
    _X_test (ndarray): 测试数据特征。
    _y_test (ndarray): 测试数据标签。
    scaler (StandardScaler): 数据标准化器。
    weights (dict): 权重参数字典。
    biases (dict): 偏置参数字典。
    train_cm (ndarray): 训练集混淆矩阵。
    test_cm (ndarray): 测试集混淆矩阵。
    """

    def __init__(self, learning_rate=0.01, lr_decay=0.95, max_iter=100, batch_size=32,
                 random_seed=42, loss=None, hidden_size=64, activation='relu'):
        """ 初始化MLP分类器

        参数：
        learning_rate (float): 学习率，默认为0.01。
        lr_decay (float): 学习率衰减系数，默认为0.95。
        max_iter (int): 最大迭代次数，默认为100。
        batch_size (int): 批量大小，默认为32。
        random_seed (int): 随机种子，默认为42。
        loss (function): 损失函数，默认为None。
        hidden_size (int): 隐藏层神经元数量，默认为64。
        activation (str): 激活函数类型，默认为'relu'。
        """
        self.learning_rate = learning_rate  # 学习率
        self.lr_decay = lr_decay  # 学习率衰减系数
        self.max_iter = max_iter  # 最大迭代次数
        self.batch_size = batch_size  # 批量大小
        self.seed = random_seed  # 随机种子
        self.loss_fn = loss  # 损失函数

        self.hidden_size = hidden_size  # 隐藏层大小
        self.activation_type = activation  # 激活函数类型

        self._X = None  # 训练数据特征
        self._y = None  # 训练数据标签
        self._X_test = None  # 测试数据特征
        self._y_test = None  # 测试数据标签

        self.scaler = StandardScaler()  # 数据标准化器
        self.weights = {}  # 权重参数
        self.biases = {}  # 偏置参数

        self.train_cm = None  # 训练集混淆矩阵
        self.test_cm = None  # 测试集混淆矩阵

    def _init_weights(self, input_dim, output_dim):
        """ 初始化网络权重参数

        参数：
        input_dim (int): 输入特征维度。
        output_dim (int): 输出类别数量。
        """
        np.random.seed(self.seed)
        # Xavier/Glorot初始化
        self.weights['w1'] = np.random.randn(input_dim, self.hidden_size) * 0.01
        self.biases['b1'] = np.zeros((1, self.hidden_size))
        self.weights['w2'] = np.random.randn(self.hidden_size, output_dim) * 0.01
        self.biases['b2'] = np.zeros((1, output_dim))

    def _relu(self, x):
        """ ReLU激活函数

        参数：
        x (ndarray): 输入数据。

        返回：
        ndarray: 激活后的输出。
        """
        return np.maximum(0, x)

    def _relu_deriv(self, x):
        """ ReLU激活函数的导数

        参数：
        x (ndarray): 输入数据。

        返回：
        ndarray: 导数结果。
        """
        return (x > 0).astype(float)

    def _softmax(self, x):
        """ Softmax函数

        参数：
        x (ndarray): 输入数据。

        返回：
        ndarray: 归一化的概率分布。
        """
        x -= np.max(x, axis=1, keepdims=True)  # 数值稳定性处理
        exps = np.exp(x)
        return exps / np.sum(exps, axis=1, keepdims=True)

    def _forward(self, X):
        """ 前向传播

        参数：
        X (ndarray): 输入特征。

        返回：
        tuple: (z1, a1, z2, a2)
            z1: 隐藏层线性变换结果
            a1: 隐藏层激活输出
            z2: 输出层线性变换结果
            a2: 输出层softmax概率
        """
        # 隐藏层计算
        z1 = np.dot(X, self.weights['w1']) + self.biases['b1']
        a1 = self._relu(z1) if self.activation_type == 'relu' else np.tanh(z1)

        # 输出层计算
        z2 = np.dot(a1, self.weights['w2']) + self.biases['b2']
        a2 = self._softmax(z2)

        return z1, a1, z2, a2

    def _backward(self, X, y_true, z1, a1, z2, a2):
        """ 反向传播计算梯度

        参数：
        X (ndarray): 输入特征。
        y_true (ndarray): 真实标签。
        z1 (ndarray): 隐藏层线性变换结果。
        a1 (ndarray): 隐藏层激活输出。
        z2 (ndarray): 输出层线性变换结果。
        a2 (ndarray): 输出层概率分布。

        返回：
        tuple: (dw1, db1, dw2, db2)
            dw1: 第一层权重梯度
            db1: 第一层偏置梯度
            dw2: 第二层权重梯度
            db2: 第二层偏置梯度
        """
        m = X.shape[0]  # 样本数量

        # 将标签转换为one-hot编码
        y_onehot = np.zeros_like(a2)
        y_onehot[np.arange(m), y_true] = 1

        # 输出层梯度
        dz2 = a2 - y_onehot
        dw2 = np.dot(a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # 隐藏层梯度
        da1 = np.dot(dz2, self.weights['w2'].T)
        if self.activation_type == 'relu':
            dz1 = da1 * self._relu_deriv(z1)
        else:
            dz1 = da1 * (1 - np.tanh(z1) ** 2)  # tanh导数
        dw1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        return dw1, db1, dw2, db2

    def _predict(self, X):
        """ 模型预测

        参数：
        X (ndarray): 输入特征。

        返回：
        tuple: (预测类别, 类别概率)
        """
        _, _, _, a2 = self._forward(X)
        return np.argmax(a2, axis=1), a2

    def train(self, X=None, y=None):
        """ 训练MLP模型

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
            raise ValueError("Must provide training data.")

        # 数据预处理
        X, y = shuffle(self._X, self._y, random_state=self.seed)
        X = self.scaler.fit_transform(X)

        # 初始化网络参数
        n_samples, n_features = X.shape
        n_classes = np.max(y) + 1
        self._init_weights(n_features, n_classes)

        # 训练循环
        for epoch in range(self.max_iter):
            X, y = shuffle(X, y)  # 每个epoch打乱数据

            # 小批量训练
            for i in range(0, n_samples, self.batch_size):
                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]

                # 前向传播
                z1, a1, z2, a2 = self._forward(X_batch)

                # 反向传播
                dw1, db1, dw2, db2 = self._backward(X_batch, y_batch, z1, a1, z2, a2)

                # 参数更新
                self.weights['w1'] -= self.learning_rate * dw1
                self.biases['b1'] -= self.learning_rate * db1
                self.weights['w2'] -= self.learning_rate * dw2
                self.biases['b2'] -= self.learning_rate * db2

            # 学习率衰减
            self.learning_rate *= self.lr_decay

        # 计算训练指标
        y_pred, y_prob = self._predict(X)
        acc = np.mean(y_pred == y)
        loss = self.loss_fn(y, y_prob)
        self.train_cm = confusion_matrix(y, y_pred)

        return acc, loss

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
            raise ValueError("Must provide test data.")

        # 数据标准化和预测
        X = self.scaler.transform(self._X_test)
        y_pred, y_prob = self._predict(X)

        # 计算评估指标
        acc = np.mean(y_pred == self._y_test)
        loss = self.loss_fn(self._y_test, y_prob)
        self.test_cm = confusion_matrix(self._y_test, y_pred)

        return acc, loss

    def predict(self, X):
        """ 对新样本进行类别预测

        参数：
        X (ndarray): 待预测数据特征。

        返回：
        ndarray: 预测的类别标签。
        """
        X = self.scaler.transform(X)
        X = np.array(X, dtype=np.float64)
        y_pred, _ = self._predict(X)
        return y_pred

    def predict_proba(self, X):
        """ 预测样本属于各个类别的概率

        参数：
        X (ndarray): 待预测数据特征。

        返回：
        ndarray: 类别概率矩阵。
        """
        X = self.scaler.transform(X)
        X = np.array(X, dtype=np.float64)
        _, a2 = self._predict(X)
        return a2

    def confusion_matrix(self):
        """ 获取混淆矩阵

        返回：
        tuple: (训练集混淆矩阵, 测试集混淆矩阵)
        """
        return self.train_cm, self.test_cm
