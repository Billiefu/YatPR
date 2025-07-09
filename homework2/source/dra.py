"""
Copyright (C) 2025 傅祉珏

:module: dra
:function: 存放数据降维方法，包括PCA和LDA
:author: 傅祉珏
:date: 2025-04-09
:license: AGPLv3 + 附加限制条款（禁止商用）

本代码依据 GNU Affero 通用公共许可证 v3 (AGPLv3) 授权，附加条款见项目根目录 ADDITIONAL_TERMS.md。
- 禁止商业用途（包括但不限于销售、集成至商业产品）
- 学术使用需在代码注释或文档中明确标注来源

AGPLv3 完整文本见 LICENSE 文件或 <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import numpy as np


def pca(X, num_components):
    """ 主成分分析（PCA）降维方法
    对输入数据进行中心化，并计算协方差矩阵，随后通过奇异值分解（SVD）获取主成分。

    参数：
    X (ndarray): 输入数据矩阵，每行是一个样本，每列是一个特征。
    num_components (int): 需要保留的主成分数目。

    返回：
    components (ndarray): 前 num_components 个主成分的矩阵。
    X_mean (ndarray): 数据的均值向量，用于中心化时的计算。
    """
    # 计算数据的均值并进行中心化
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # 计算协方差矩阵
    cov_matrix = np.cov(X_centered, rowvar=False)

    # 进行奇异值分解（SVD）
    U, S, Vt = np.linalg.svd(cov_matrix)

    # 提取前 num_components 个主成分
    components = Vt[:num_components]

    return components, X_mean


def lda(X, y, num_components):
    """ 线性判别分析（LDA）降维方法
    通过最大化类间散度与类内散度的比值来找到最优的投影方向。

    参数：
    X (ndarray): 输入数据矩阵，每行是一个样本，每列是一个特征。
    y (ndarray): 样本的标签，用于计算类内散度和类间散度。
    num_components (int): 需要保留的线性判别成分数目。

    返回：
    components (ndarray): 前 num_components 个线性判别成分的矩阵。
    """
    # 获取类别标签
    class_labels = np.unique(y)

    # 计算整体均值
    mean_overall = np.mean(X, axis=0)

    # 初始化类内散度矩阵和类间散度矩阵
    Sw = np.zeros((X.shape[1], X.shape[1]))
    Sb = np.zeros((X.shape[1], X.shape[1]))

    # 计算每一类的类内散度和类间散度
    for c in class_labels:
        # 获取当前类别的样本
        X_c = X[y == c]

        # 计算该类的均值
        mean_c = np.mean(X_c, axis=0)

        # 类内散度的计算
        Sw += np.dot((X_c - mean_c).T, (X_c - mean_c))

        # 计算类间散度
        n_c = X_c.shape[0]
        mean_diff = (mean_c - mean_overall).reshape(-1, 1)
        Sb += n_c * np.dot(mean_diff, mean_diff.T)

    # 计算广义瑞利商的特征值和特征向量
    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(Sw).dot(Sb))

    # 按特征值降序排序，并选择前 num_components 个特征向量作为最终的投影方向
    sorted_indices = np.argsort(-eigvals.real)
    components = eigvecs[:, sorted_indices[:num_components]].real

    return components
