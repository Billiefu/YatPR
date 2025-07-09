"""
Copyright (C) 2025 傅祉珏 杨程骏 谢敬豪

:module: loss
:function: 用于存放损失函数
:author: 傅祉珏，杨程骏，谢敬豪
:date: 2025-07-03
:license: AGPLv3 + 附加限制条款（禁止商用）

本代码依据 GNU Affero 通用公共许可证 v3 (AGPLv3) 授权，附加条款见项目根目录 ADDITIONAL_TERMS.md。
- 禁止商业用途（包括但不限于销售、集成至商业产品）
- 学术使用需在代码注释或文档中明确标注来源

AGPLv3 完整文本见 LICENSE 文件或 <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import numpy as np


class CrossEntropyLoss:
    """ 多分类交叉熵损失函数

    适用于多分类任务的概率输出模型，常用于神经网络分类器。
    """

    def __call__(self, y_true, y_pred_proba):
        """ 计算交叉熵损失

        参数：
        y_true (ndarray): 真实类别标签，shape=(n_samples,)。
        y_pred_proba (ndarray): 预测类别概率，shape=(n_samples, n_classes)。

        返回：
        float: 交叉熵损失值。
        """
        eps = 1e-12  # 防止数值溢出的极小值
        y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)  # 裁剪概率值到合理范围

        n_samples = y_true.shape[0]
        # 将真实标签转换为one-hot编码
        y_onehot = np.zeros_like(y_pred_proba)
        y_onehot[np.arange(n_samples), y_true] = 1

        # 计算交叉熵损失
        loss = -np.sum(y_onehot * np.log(y_pred_proba)) / n_samples
        return loss


class MultiClassHingeLoss:
    """ 多分类铰链损失函数

    适用于支持向量机等多分类任务，最大化分类间隔。
    """

    def __call__(self, y_true, scores):
        """ 计算多分类铰链损失

        参数：
        y_true (ndarray): 真实类别标签，shape=(n_samples,)。
        scores (ndarray): 分类器输出的原始得分，shape=(n_samples, n_classes)。

        返回：
        float: 铰链损失值。
        """
        n_samples = scores.shape[0]
        # 获取正确类别的得分
        correct_class_scores = scores[np.arange(n_samples), y_true].reshape(-1, 1)
        # 计算间隔(margin)，delta=1
        margins = np.maximum(0, scores - correct_class_scores + 1)
        # 忽略正确类别的margin
        margins[np.arange(n_samples), y_true] = 0
        # 计算平均损失
        loss = np.sum(margins) / n_samples
        return loss


class ImpurityEntropyLoss:
    """ 信息熵损失函数

    用于衡量类别分布的不纯度，常用于决策树等算法。
    """

    def __call__(self, y):
        """ 计算信息熵

        参数：
        y (ndarray): 类别标签数组，shape=(n_samples,)。

        返回：
        float: 信息熵值。
        """
        _, counts = np.unique(y, return_counts=True)  # 统计各类别数量
        probs = counts / len(y)  # 计算各类别概率
        return -np.sum(probs * np.log2(probs + 1e-12))  # 计算信息熵


class AccuracyLoss:
    """ 准确率损失函数

    将准确率转换为损失值形式(1-accuracy)，便于统一接口。
    """

    def __call__(self, y_true, y_pred):
        """ 计算准确率损失

        参数：
        y_true (ndarray): 真实类别标签。
        y_pred (ndarray): 预测类别标签。

        返回：
        float: 1 - 准确率。
        """
        return 1.0 - np.mean(y_true == y_pred)


class ClusteringLoss:
    """ 聚类损失函数

    用于聚类任务，衡量聚类结果与真实标签的差异。
    """

    def __call__(self, y_true, y_pred):
        """ 计算聚类损失

        参数：
        y_true (ndarray): 真实类别标签。
        y_pred (ndarray): 预测类别标签。

        返回：
        float: 1 - 聚类准确率。
        """
        return 1.0 - np.mean(y_true == y_pred)


# 以下为损失函数的工厂函数，便于统一调用接口

def cross_entropy():
    """ 创建交叉熵损失函数实例

    返回：
    CrossEntropyLoss: 交叉熵损失函数实例。
    """
    return CrossEntropyLoss()


def multi_class_hinge():
    """ 创建多分类铰链损失函数实例

    返回：
    MultiClassHingeLoss: 多分类铰链损失函数实例。
    """
    return MultiClassHingeLoss()


def impurity_entropy():
    """ 创建信息熵损失函数实例

    返回：
    ImpurityEntropyLoss: 信息熵损失函数实例。
    """
    return ImpurityEntropyLoss()


def accuracy():
    """ 创建准确率损失函数实例

    返回：
    AccuracyLoss: 准确率损失函数实例。
    """
    return AccuracyLoss()


def clustering_loss():
    """ 创建聚类损失函数实例

    返回：
    ClusteringLoss: 聚类损失函数实例。
    """
    return ClusteringLoss()
