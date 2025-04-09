"""
@module test
@function 用于进行人脸识别的任务
@author 傅祉珏
@date 2025年4月9日
"""

import numpy as np
from scipy import io
from sklearn.preprocessing import StandardScaler

import utils
from dra import pca, lda
from knn import KNN
from svm import SVM

# 加载数据
x = io.loadmat('Yale_64x64.mat')  # 从.mat文件中加载人脸数据
ins_perclass, class_number, train_test_split = 11, 15, 9  # 每类样本数、类数和训练集与测试集的分割比例
input_dim = x['fea'].shape[1]  # 输入特征的维度

# 重塑数据
feat = x['fea'].reshape(-1, ins_perclass, input_dim)  # 将特征矩阵重塑为每类样本的数组
label = x['gnd'].reshape(-1, ins_perclass)  # 将标签重塑为每类样本的标签
train_data = feat[:, :train_test_split, :].reshape(-1, input_dim)  # 获取训练集特征
test_data = feat[:, train_test_split:, :].reshape(-1, input_dim)  # 获取测试集特征
train_label = label[:, :train_test_split].reshape(-1)  # 获取训练集标签
test_label = label[:, train_test_split:].reshape(-1)  # 获取测试集标签

# 进行PCA操作
pca_components, pca_mean = pca(train_data, 8)  # 使用PCA降维，保留8个主成分
utils.show_vectors_as_images(pca_components, title="Top 8 PCA Eigenfaces")  # 展示前8个PCA特征图像
train_pca = (train_data - pca_mean) @ pca_components.T  # 将训练数据投影到PCA空间
test_pca = (test_data - pca_mean) @ pca_components.T  # 将测试数据投影到PCA空间
utils.visualize_2d(train_pca, train_label, "PCA - Train (2D)")  # 可视化PCA降维后的训练集
utils.visualize_2d(test_pca, test_label, "PCA - Test (2D)")  # 可视化PCA降维后的测试集

# 进行LDA操作
lda_components = lda(train_data, train_label, 8)  # 使用LDA降维，保留8个判别成分
utils.show_vectors_as_images(lda_components.T, title="Top 8 LDA Discriminants")  # 展示前8个LDA判别特征图像
train_lda = train_data @ lda_components  # 将训练数据投影到LDA空间
test_lda = test_data @ lda_components  # 将测试数据投影到LDA空间
utils.visualize_2d(train_lda, train_label, "LDA - Train (2D)")  # 可视化LDA降维后的训练集
utils.visualize_2d(test_lda, test_label, "LDA - Test (2D)")  # 可视化LDA降维后的测试集

# 数据标准化
scaler = StandardScaler()  # 初始化标准化器

train_pca = scaler.fit_transform(train_pca)  # 对PCA降维后的训练数据进行标准化
test_pca = scaler.transform(test_pca)  # 对PCA降维后的测试数据进行标准化
utils.visualize_2d(train_pca, train_label, "PCA - Train (2D)")  # 可视化标准化后的PCA训练集
utils.visualize_2d(test_pca, test_label, "PCA - Test (2D)")  # 可视化标准化后的PCA测试集

train_lda = scaler.fit_transform(train_lda)  # 对LDA降维后的训练数据进行标准化
test_lda = scaler.transform(test_lda)  # 对LDA降维后的测试数据进行标准化
utils.visualize_2d(train_lda, train_label, "LDA - Train (2D)")  # 可视化标准化后的LDA训练集
utils.visualize_2d(test_lda, test_label, "LDA - Test (2D)")  # 可视化标准化后的LDA测试集

# KNN分类器分类
knn = KNN(k=3)  # 初始化KNN分类器，k=3

knn.fit(train_pca, train_label)  # 使用PCA降维后的训练集进行训练
accuracy = knn.score(test_pca, test_label)  # 在PCA降维后的测试集上进行预测并计算准确率
print(f"PCA + KNN Accuracy: {accuracy * 100 :.2f}%")  # 输出PCA + KNN分类的准确率

knn.fit(train_lda, train_label)  # 使用LDA降维后的训练集进行训练
accuracy = knn.score(test_lda, test_label)  # 在LDA降维后的测试集上进行预测并计算准确率
print(f"LDA + KNN Accuracy: {accuracy * 100 :.2f}%")  # 输出LDA + KNN分类的准确率

# SVM分类器分类
svm = SVM(kernel='rbf', C=10.0, gamma=0.1)  # 初始化SVM分类器，使用RBF核

svm.fit(train_pca, train_label)  # 使用PCA降维后的训练集进行训练
y_pred = svm.predict(test_pca)  # 在PCA降维后的测试集上进行预测
acc = np.mean(y_pred == test_label)  # 计算PCA + SVM分类的准确率
print(f"PCA + SVM Accuracy: {acc * 100 :.2f}%")  # 输出PCA + SVM分类的准确率

svm.fit(train_lda, train_label)  # 使用LDA降维后的训练集进行训练
y_pred = svm.predict(test_lda)  # 在LDA降维后的测试集上进行预测
acc = np.mean(y_pred == test_label)  # 计算LDA + SVM分类的准确率
print(f"LDA + SVM Accuracy: {acc * 100 :.2f}%")  # 输出LDA + SVM分类的准确率
