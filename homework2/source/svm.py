"""
@module svm
@function 存放SVM分类器
@author 傅祉珏
@date 2025年4月9日
"""

import numpy as np
from cvxopt import matrix, solvers


def linear_kernel(x, y):
    """ 线性核函数

    参数：
    x (ndarray): 输入数据向量。
    y (ndarray): 输入数据向量。

    返回：
    float: 线性核函数计算结果。
    """
    return np.dot(x, y.T)


def polynomial_kernel(x, y, degree=3, coef0=1):
    """ 多项式核函数

    参数：
    x (ndarray): 输入数据向量。
    y (ndarray): 输入数据向量。
    degree (int): 多项式的度数，默认为3。
    coef0 (float): 核函数的常数项，默认为1。

    返回：
    float: 多项式核函数计算结果。
    """
    return (np.dot(x, y.T) + coef0) ** degree


def rbf_kernel(x, y, gamma=0.05):
    """ 径向基核函数（RBF）

    参数：
    x (ndarray): 输入数据向量。
    y (ndarray): 输入数据向量。
    gamma (float): 径向基函数的参数，默认为0.05。

    返回：
    float: 径向基核函数计算结果。
    """
    if x.ndim == 1:
        x = x[np.newaxis, :]
    if y.ndim == 1:
        y = y[np.newaxis, :]
    dist = np.sum((x[:, np.newaxis] - y) ** 2, axis=2)
    return np.exp(-gamma * dist)


def cosine_kernel(x, y):
    """ 余弦相似度核函数

    参数：
    x (ndarray): 输入数据向量。
    y (ndarray): 输入数据向量。

    返回：
    float: 余弦相似度核函数计算结果。
    """
    x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)
    y_norm = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-10)
    return np.dot(x_norm, y_norm.T)


def sigmoid_kernel(x, y, alpha=0.01, coef0=1):
    """ Sigmoid核函数

    参数：
    x (ndarray): 输入数据向量。
    y (ndarray): 输入数据向量。
    alpha (float): 核函数的参数，默认为0.01。
    coef0 (float): 核函数的常数项，默认为1。

    返回：
    float: Sigmoid核函数计算结果。
    """
    return np.tanh(alpha * np.dot(x, y.T) + coef0)


class SVM:
    """ 支持向量机（SVM）分类器类
    该类实现了SVM分类算法，包括训练、预测和支持向量机的相关操作。

    属性：
    C (float): 正则化参数，控制软间隔的宽度。
    kernel_type (str): 核函数类型，支持'linear'、'poly'、'rbf'、'cosine'、'sigmoid'等。
    kernel_params (dict): 核函数的附加参数，如多项式的度数、RBF的gamma值等。
    models (dict): 用于存储每一类的训练模型，支持一对多的多分类任务。
    """

    def __init__(self, kernel='linear', C=1.0, **kwargs):
        """ 初始化SVM分类器

        参数：
        kernel (str): 核函数类型，默认为'linear'。
        C (float): 正则化参数，默认为1.0。
        kwargs (dict): 其他核函数的附加参数。
        """
        self.C = C  # 正则化参数
        self.kernel_type = kernel  # 核函数类型
        self.kernel_params = kwargs  # 核函数的附加参数
        self.models = {}  # 存储每个类别的训练模型

    def _compute_kernel(self, X1, X2):
        """ 根据设定的核函数计算样本之间的核矩阵

        参数：
        X1 (ndarray): 样本矩阵1。
        X2 (ndarray): 样本矩阵2。

        返回：
        ndarray: 计算得到的核矩阵。
        """
        if self.kernel_type == 'linear':
            return linear_kernel(X1, X2)
        elif self.kernel_type == 'poly':
            return polynomial_kernel(X1, X2, **self.kernel_params)
        elif self.kernel_type == 'rbf':
            return rbf_kernel(X1, X2, **self.kernel_params)
        elif self.kernel_type == 'cosine':
            return cosine_kernel(X1, X2)
        elif self.kernel_type == 'sigmoid':
            return sigmoid_kernel(X1, X2, **self.kernel_params)
        else:
            raise ValueError("Unknown kernel type:", self.kernel_type)

    def _train_binary(self, X, y):
        """ 训练二分类SVM模型

        参数：
        X (ndarray): 训练数据特征矩阵。
        y (ndarray): 训练数据的标签，值为1或-1。
        """
        n_samples = X.shape[0]

        # 如果所有样本都属于一个类别，跳过该类别的训练
        if np.all(y == 1) or np.all(y == -1):
            print("All samples belong to one class, skipping training for this class.")
            self.support_vectors = None
            return

        # 计算核矩阵
        K = self._compute_kernel(X, X)
        P = np.outer(y, y) * K
        P = 0.5 * (P + P.T)  # 保证核矩阵对称
        P += 1e-6 * np.eye(n_samples)  # 防止奇异矩阵
        P = matrix(P)

        # 定义优化问题的目标函数和约束
        q = matrix(-np.ones(n_samples))
        G_std = -np.eye(n_samples)
        h_std = np.zeros(n_samples)
        G_slack = np.eye(n_samples)
        h_slack = np.ones(n_samples) * self.C
        G = matrix(np.vstack((G_std, G_slack)))
        h = matrix(np.hstack((h_std, h_slack)))

        A = matrix(y.reshape(1, -1).astype('double'))
        b = matrix(np.zeros(1))

        solvers.options['show_progress'] = False

        # 求解二次规划问题
        try:
            solution = solvers.qp(P, q, G, h, A, b)
        except ValueError as e:
            print("Solver failed:", e)
            self.support_vectors = None
            return

        # 获取拉格朗日乘子alpha
        alphas = np.ravel(solution['x'])

        # 确定支持向量
        sv = alphas > 1e-5
        self.alphas = alphas[sv]
        self.support_vectors = X[sv]
        self.support_y = y[sv]
        self.sv_indices = np.where(sv)[0]

        # 计算偏置项b
        self.kernel_sv = self._compute_kernel(self.support_vectors, self.support_vectors)
        self.b = np.mean([
            y_k - np.sum(self.alphas * self.support_y * self._compute_kernel(X_k[np.newaxis, :], self.support_vectors))
            for X_k, y_k in zip(self.support_vectors, self.support_y)
        ])

    def fit(self, X, y):
        """ 训练SVM分类器

        参数：
        X (ndarray): 训练数据特征矩阵。
        y (ndarray): 训练数据的标签。
        """
        self.classes = np.unique(y)  # 获取所有类别
        for cls in self.classes:
            binary_y = np.where(y == cls, 1, -1)  # 转换为二分类标签
            model = SVM(kernel=self.kernel_type, C=self.C, **self.kernel_params)
            model._train_binary(X, binary_y)  # 训练二分类模型

            if model.support_vectors is not None:
                self.models[cls] = model  # 存储训练好的模型
            else:
                print(f"Model for class {cls} did not train successfully.")

    def predict(self, X):
        """ 对新样本进行预测

        参数：
        X (ndarray): 待预测的数据特征矩阵。

        返回：
        ndarray: 每个样本的预测类别标签。
        """
        scores = []
        for cls in self.classes:
            model = self.models[cls]
            if model.support_vectors is None:
                continue
            K = model._compute_kernel(X, model.support_vectors)
            score = np.dot(K, model.alphas * model.support_y) + model.b
            scores.append(score)
        return self.classes[np.argmax(np.vstack(scores), axis=0)]  # 返回预测类别
