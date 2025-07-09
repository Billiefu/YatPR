"""
Copyright (C) 2025 傅祉珏

:module: ensemble
:function: 用于完成模型集成的任务
:author: 傅祉珏
:date: 2025-07-03
:license: AGPLv3 + 附加限制条款（禁止商用）

本代码依据 GNU Affero 通用公共许可证 v3 (AGPLv3) 授权，附加条款见项目根目录 ADDITIONAL_TERMS.md。
- 禁止商业用途（包括但不限于销售、集成至商业产品）
- 学术使用需在代码注释或文档中明确标注来源

AGPLv3 完整文本见 LICENSE 文件或 <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import numpy as np


class StackingEnsemble:
    """ 堆叠(Stacking)集成学习模型

    属性：
    base_models (list): 基学习器列表。
    meta_model (object): 元学习器。
    use_proba (bool): 是否使用概率预测作为特征。
    """

    def __init__(self, base_models, meta_model, use_proba=True):
        """ 初始化堆叠集成模型

        参数：
        base_models (list): 基学习器列表。
        meta_model (object): 元学习器。
        use_proba (bool): 是否使用概率预测作为特征，默认为True。
        """
        self.base_models = base_models  # 基学习器列表
        self.meta_model = meta_model  # 元学习器
        self.use_proba = use_proba  # 是否使用概率预测

    def _get_model_output(self, model, X):
        """ 获取基学习器的输出特征

        参数：
        model (object): 基学习器。
        X (ndarray): 输入特征矩阵。

        返回：
        ndarray: 基学习器的输出特征。
        """
        if self.use_proba and hasattr(model, "predict_proba"):
            return model.predict_proba(X)  # 使用概率预测
        else:
            return model.predict(X).reshape(-1, 1)  # 使用类别预测并调整形状

    def _stack_features(self, X):
        """ 堆叠基学习器的输出特征

        参数：
        X (ndarray): 输入特征矩阵。

        返回：
        ndarray: 堆叠后的特征矩阵。
        """
        outputs = []
        for i, model in enumerate(self.base_models):
            proba = model.predict_proba(X)  # 获取每个基学习器的概率预测
            outputs.append(proba)
        return np.hstack(outputs)  # 水平堆叠所有基学习器的输出

    def train(self, X_train, y_train):
        """ 训练堆叠集成模型

        参数：
        X_train (ndarray): 训练数据特征。
        y_train (ndarray): 训练数据标签。

        返回：
        tuple: (训练准确率, 训练损失)
        """
        meta_X = self._stack_features(X_train)  # 生成元特征
        train_acc, train_loss = self.meta_model.train(meta_X, y_train)  # 训练元学习器
        return train_acc, train_loss

    def evaluate(self, X_test, y_test):
        """ 评估堆叠集成模型

        参数：
        X_test (ndarray): 测试数据特征。
        y_test (ndarray): 测试数据标签。

        返回：
        tuple: (测试准确率, 测试损失)
        """
        meta_X = self._stack_features(X_test)  # 生成元特征
        test_acc, test_loss = self.meta_model.evaluate(meta_X, y_test)  # 评估元学习器
        return test_acc, test_loss

    def confusion_matrix(self):
        """ 获取混淆矩阵

        返回：
        tuple: (训练集混淆矩阵, 测试集混淆矩阵)
        """
        train_cm, test_cm = self.meta_model.confusion_matrix()
        return train_cm, test_cm
