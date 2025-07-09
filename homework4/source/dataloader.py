"""
Copyright (C) 2025 傅祉珏 杨程骏

:module: dataloader
:function: 用于数据预处理及数据集划分
:author: 傅祉珏，杨程骏
:date: 2025-07-03
:license: AGPLv3 + 附加限制条款（禁止商用）

本代码依据 GNU Affero 通用公共许可证 v3 (AGPLv3) 授权，附加条款见项目根目录 ADDITIONAL_TERMS.md。
- 禁止商业用途（包括但不限于销售、集成至商业产品）
- 学术使用需在代码注释或文档中明确标注来源

AGPLv3 完整文本见 LICENSE 文件或 <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class OULADDataLoader:
    """ OULAD数据集加载与预处理类

    属性：
    data_dir (str): 数据文件目录路径。
    random_state (int): 随机种子。
    data (dict): 存储原始数据的字典。
    features (DataFrame): 预处理后的特征数据。
    labels (Series): 预处理后的标签数据。
    encoders (dict): 存储标签编码器的字典。
    """

    def __init__(self, data_dir='./data', random_state=42):
        """ 初始化数据加载器

        参数：
        data_dir (str): 数据目录路径，默认为'./data'。
        random_state (int): 随机种子，默认为42。
        """
        self.data_dir = data_dir  # 数据目录路径
        self.random_state = random_state  # 随机种子
        self.data = {}  # 原始数据存储字典
        self.features = None  # 特征数据
        self.labels = None  # 标签数据
        self.encoders = {}  # 标签编码器字典

    def load_raw_data(self):
        """ 加载所有原始CSV数据集

        从指定目录加载OULAD数据集的所有CSV文件，包括：
        - assessments.csv: 评估信息
        - courses.csv: 课程信息
        - studentAssessment.csv: 学生评估记录
        - studentVle.csv: 学生虚拟学习环境活动
        - studentInfo.csv: 学生基本信息（可选）
        - vle.csv: 虚拟学习环境信息（可选）
        """
        # 核心数据集
        self.data['assessments'] = pd.read_csv(f'{self.data_dir}/assessments.csv')
        self.data['courses'] = pd.read_csv(f'{self.data_dir}/courses.csv')
        self.data['studentAssessment'] = pd.read_csv(f'{self.data_dir}/studentAssessment.csv')
        self.data['studentVle'] = pd.read_csv(f'{self.data_dir}/studentVle.csv')

        # 可选数据集
        if os.path.exists(f'{self.data_dir}/studentInfo.csv'):
            self.data['studentInfo'] = pd.read_csv(f'{self.data_dir}/studentInfo.csv')
        if os.path.exists(f'{self.data_dir}/vle.csv'):
            self.data['vle'] = pd.read_csv(f'{self.data_dir}/vle.csv')

    def _convert_to_numeric(self, df, column_name):
        """ 将分类变量转换为数值编码

        参数：
        df (DataFrame): 待处理的数据框。
        column_name (str): 需要转换的列名。

        返回：
        DataFrame: 转换后的数据框。
        """
        if column_name not in df.columns:
            return df.copy()

        # 初始化或复用标签编码器
        if column_name not in self.encoders:
            self.encoders[column_name] = LabelEncoder()
            self.encoders[column_name].fit(df[column_name].astype(str).fillna("MISSING"))

        df = df.copy()
        # 执行标签编码转换
        encoded_values = self.encoders[column_name].transform(
            df[column_name].astype(str).fillna("MISSING"))
        df.loc[:, column_name] = pd.to_numeric(encoded_values, errors='coerce')
        return df

    def preprocess_data(self):
        """ 数据预处理和特征工程主函数

        执行以下操作：
        1. 合并评估和课程数据
        2. 计算学生表现指标
        3. 计算VLE参与度指标
        4. 合并学生基本信息（如有）
        5. 生成衍生特征
        6. 处理缺失值
        7. 分类变量编码
        8. 标签编码
        """
        # 1. 合并评估和课程数据
        assessments_courses = pd.merge(
            self.data['assessments'],
            self.data['courses'],
            on=['code_module', 'code_presentation']
        )

        # 2. 合并学生评估数据
        student_assessments = pd.merge(
            self.data['studentAssessment'],
            assessments_courses,
            on='id_assessment'
        )

        # 3. 计算学生表现指标
        student_performance = student_assessments.groupby(
            ['id_student', 'code_module', 'code_presentation']
        ).agg({
            'score': ['mean', 'count'],
            'date_submitted': 'mean',
            'weight': 'sum'
        }).reset_index()
        student_performance.columns = ['id_student', 'code_module', 'code_presentation',
                                       'avg_score', 'assessment_count', 'avg_submit_day', 'total_weight']

        # 4. 计算VLE参与度指标
        vle_engagement = self.data['studentVle'].groupby(
            ['id_student', 'code_module', 'code_presentation']
        ).agg({
            'sum_click': ['sum', 'mean', 'count'],
            'date': ['min', 'max']
        }).reset_index()
        vle_engagement.columns = ['id_student', 'code_module', 'code_presentation',
                                  'total_clicks', 'avg_clicks', 'activity_count',
                                  'first_activity_day', 'last_activity_day']

        # 5. 合并特征数据
        student_features = pd.merge(
            student_performance,
            vle_engagement,
            on=['id_student', 'code_module', 'code_presentation'],
            how='left'
        )

        # 6. 合并学生基本信息（如有）
        if 'studentInfo' in self.data:
            student_features = pd.merge(
                student_features,
                self.data['studentInfo'],
                on=['id_student', 'code_module', 'code_presentation'],
                how='left'
            )

        # 7. 特征工程 - 生成衍生特征
        student_features['activity_duration'] = student_features['last_activity_day'] - student_features[
            'first_activity_day']
        student_features['clicks_per_day'] = student_features['total_clicks'] / student_features[
            'activity_duration'].clip(lower=1)

        # 8. 处理数值型缺失值
        numeric_cols = student_features.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if student_features[col].isnull().any():
                # 根据数据偏斜程度选择填充策略
                if abs(student_features[col].skew()) > 1:
                    fill_value = student_features[col].median()
                else:
                    fill_value = student_features[col].mean()
                student_features.loc[:, col] = student_features[col].fillna(fill_value)

        # 9. 处理分类型缺失值
        categorical_cols = student_features.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if student_features[col].isnull().any():
                mode_value = student_features[col].mode()[0]
                student_features.loc[:, col] = student_features[col].fillna(mode_value)

        # 10. 分类变量编码
        categorical_cols = ['code_module', 'code_presentation']
        optional_cols = ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']
        for col in optional_cols:
            if col in student_features.columns:
                categorical_cols.append(col)

        for col in categorical_cols:
            student_features = self._convert_to_numeric(student_features, col)

        self.features = student_features

        # 11. 标签编码处理
        if 'final_result' in student_features.columns:
            if student_features['final_result'].dtype == 'object':
                self.labels = self._convert_to_numeric(student_features[['final_result']], 'final_result')[
                    'final_result']
            else:
                self.labels = student_features['final_result']
            self.features = student_features.drop('final_result', axis=1)

        # 打印标签映射关系
        print("Label mapping for final_result:")
        for i, label in enumerate(self.encoders['final_result'].classes_):
            print(f"Class {i}: {label}")
        print()

        # 12. 强制类型转换和最终缺失值处理
        numeric_cols = self.features.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            self.features.loc[:, col] = pd.to_numeric(self.features[col], errors='coerce')

        if self.labels is not None:
            self.labels = pd.to_numeric(self.labels, errors='coerce')

        # 最终缺失值填充
        self.features.fillna(0, inplace=True)
        if self.labels is not None:
            self.labels.fillna(0, inplace=True)

    def split_data(self, test_size=0.2, val_size=0.1):
        """ 划分训练集、验证集和测试集

        参数：
        test_size (float): 测试集比例，默认为0.2。
        val_size (float): 验证集比例，默认为0.1。

        返回：
        tuple: (X_train, X_test, y_train, y_test)
        """
        if self.features is None:
            self.load_raw_data()
            self.preprocess_data()

        # 使用分层抽样划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            self.features,
            self.labels if self.labels is not None else np.zeros(len(self.features)),
            test_size=test_size,
            random_state=self.random_state,
            stratify=self.labels if self.labels is not None else None
        )

        return X_train, X_test, y_train, y_test
