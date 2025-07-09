"""
Copyright (C) 2025 傅祉珏 杨程骏

:module: cart
:function: 存放决策树分类器
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


class TreeNode:
    """ 决策树节点类

    属性：
    depth (int): 当前节点在树中的深度。
    feature_index (int): 用于分割的特征索引。
    threshold (float): 分割特征的阈值。
    left (TreeNode): 左子节点。
    right (TreeNode): 右子节点。
    label (int): 叶节点的类别标签。
    """

    def __init__(self, depth=0):
        """ 初始化决策树节点

        参数：
        depth (int): 节点的初始深度，默认为0。
        """
        self.depth = depth
        self.feature_index = None  # 分割特征索引
        self.threshold = None  # 分割阈值
        self.left = None  # 左子树
        self.right = None  # 右子树
        self.label = None  # 叶节点类别标签


class CART:
    """ CART(Classification and Regression Trees)决策树分类器

    属性：
    cm (ndarray): 混淆矩阵。
    num_classes (int): 类别数量。
    loss_fn (function): 损失函数。
    max_depth (int): 树的最大深度。
    min_samples_split (int): 节点分裂的最小样本数。
    root (TreeNode): 决策树的根节点。
    _X (ndarray): 训练数据特征。
    _y (ndarray): 训练数据标签。
    _X_test (ndarray): 测试数据特征。
    _y_test (ndarray): 测试数据标签。
    scaler (StandardScaler): 数据标准化器。

    (以下参数仅为接口统一保留，实际不使用)
    learning_rate (float): 学习率。
    lr_decay (float): 学习率衰减系数。
    max_iter (int): 最大迭代次数。
    batch_size (int): 批量大小。
    random_seed (int): 随机种子。
    """

    def __init__(self, learning_rate=0.01, lr_decay=0.95, max_iter=1, batch_size=32,
                 random_seed=42, loss=None, max_depth=7, min_samples_split=2):
        """ 初始化CART分类器

        参数：
        learning_rate (float): 学习率(保留参数)，默认为0.01。
        lr_decay (float): 学习率衰减系数(保留参数)，默认为0.95。
        max_iter (int): 最大迭代次数(保留参数)，默认为1。
        batch_size (int): 批量大小(保留参数)，默认为32。
        random_seed (int): 随机种子，默认为42。
        loss (function): 损失函数，默认为None。
        max_depth (int): 树的最大深度，默认为7。
        min_samples_split (int): 节点分裂的最小样本数，默认为2。
        """
        self.cm = None  # 混淆矩阵
        self.num_classes = None  # 类别数量
        self.loss_fn = loss  # 损失函数
        self.max_depth = max_depth  # 树的最大深度
        self.min_samples_split = min_samples_split  # 最小分裂样本数
        self.root = None  # 决策树根节点
        self._X = None  # 训练数据特征
        self._y = None  # 训练数据标签
        self._X_test = None  # 测试数据特征
        self._y_test = None  # 测试数据标签
        self.scaler = StandardScaler()  # 数据标准化器

        # 以下参数仅为接口统一保留，实际不使用
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_seed = random_seed

        self.train_cm = None  # 训练集混淆矩阵
        self.test_cm = None  # 测试集混淆矩阵

    def _gini_impurity(self, y):
        """ 计算基尼不纯度

        参数：
        y (ndarray): 样本标签数组。

        返回：
        float: 基尼不纯度值。
        """
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def _best_split(self, X, y):
        """ 寻找最佳分割特征和阈值

        参数：
        X (ndarray): 样本特征矩阵。
        y (ndarray): 样本标签数组。

        返回：
        tuple: (最佳特征索引, 最佳分割阈值)
        """
        best_gain = -1  # 最佳信息增益
        best_feature = None  # 最佳特征索引
        best_threshold = None  # 最佳分割阈值
        current_impurity = self.loss_fn(y)  # 当前节点的不纯度

        n_samples, n_features = X.shape

        # 遍历所有特征寻找最佳分割
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_idx = X[:, feature_index] <= threshold
                right_idx = ~left_idx

                # 检查分裂后的样本数是否满足最小要求
                if np.sum(left_idx) < self.min_samples_split or np.sum(right_idx) < self.min_samples_split:
                    continue

                # 计算分裂后的不纯度
                left_impurity = self.loss_fn(y[left_idx])
                right_impurity = self.loss_fn(y[right_idx])
                gain = current_impurity - (
                        (np.sum(left_idx) * left_impurity + np.sum(right_idx) * right_impurity) / len(y)
                )

                # 更新最佳分割
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth):
        """ 递归构建决策树

        参数：
        X (ndarray): 样本特征矩阵。
        y (ndarray): 样本标签数组。
        depth (int): 当前节点深度。

        返回：
        TreeNode: 构建完成的子树根节点。
        """
        node = TreeNode(depth=depth)

        # 终止条件：纯节点/达到最大深度/样本数不足
        if len(set(y)) == 1 or depth >= self.max_depth or len(y) < self.min_samples_split:
            node.label = np.bincount(y).argmax()  # 设置为叶节点
            return node

        # 寻找最佳分割
        feature_index, threshold = self._best_split(X, y)

        # 无法找到有效分割时设置为叶节点
        if feature_index is None:
            node.label = np.bincount(y).argmax()
            return node

        # 设置节点分割属性
        node.feature_index = feature_index
        node.threshold = threshold

        # 递归构建左右子树
        left_idx = X[:, feature_index] <= threshold
        right_idx = ~left_idx

        node.left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        node.right = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return node

    def _predict_one(self, x, node):
        """ 对单个样本进行预测

        参数：
        x (ndarray): 单个样本特征。
        node (TreeNode): 当前决策树节点。

        返回：
        int: 预测的类别标签。
        """
        while node.label is None:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.label

    def _predict(self, X):
        """ 对多个样本进行预测

        参数：
        X (ndarray): 样本特征矩阵。

        返回：
        ndarray: 预测的类别标签数组。
        """
        return np.array([self._predict_one(x, self.root) for x in X])

    def train(self, X=None, y=None):
        """ 训练决策树模型

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
            self.num_classes = len(np.unique(y))

        if self._X is None or self._y is None:
            raise ValueError("Must provide training data to train().")

        # 数据预处理
        np.random.seed(self.random_seed)
        X, y = shuffle(self._X, self._y, random_state=self.random_seed)
        X = self.scaler.fit_transform(X)

        # 构建决策树
        self.root = self._build_tree(X, y, depth=0)

        # 评估训练结果
        y_pred = self._predict(X)
        acc = np.mean(y_pred == y)
        loss = self.loss_fn(y)
        self.train_cm = confusion_matrix(y, y_pred)

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

        # 数据预处理和预测
        X = self.scaler.transform(self._X_test)
        y_pred = self._predict(X)

        # 计算评估指标
        acc = np.mean(y_pred == self._y_test)
        loss = self.loss_fn(self._y_test)
        self.test_cm = confusion_matrix(self._y_test, y_pred)

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
        """ 预测样本属于各个类别的概率

        参数：
        X (ndarray): 待预测数据特征。

        返回：
        ndarray: 概率矩阵，shape=(n_samples, n_classes)。
        """
        X = self.scaler.transform(X)
        X = np.array(X, dtype=np.float64)
        y_pred = self._predict(X)
        proba = np.zeros((len(y_pred), self.num_classes))
        proba[np.arange(len(y_pred)), y_pred] = 1
        return proba

    def confusion_matrix(self):
        """ 获取混淆矩阵

        返回：
        tuple: (训练集混淆矩阵, 测试集混淆矩阵)
        """
        return self.train_cm, self.test_cm
