"""
Copyright (C) 2025 傅祉珏

:module: utils
:function: 存放所有可能需要用到的共性函数
:author: 傅祉珏
:date: 2025-04-09
:license: AGPLv3 + 附加限制条款（禁止商用）

本代码依据 GNU Affero 通用公共许可证 v3 (AGPLv3) 授权，附加条款见项目根目录 ADDITIONAL_TERMS.md。
- 禁止商业用途（包括但不限于销售、集成至商业产品）
- 学术使用需在代码注释或文档中明确标注来源

AGPLv3 完整文本见 LICENSE 文件或 <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def show_vectors_as_images(vectors, image_shape=(64, 64), num_show=8, title=""):
    """ 将向量可视化为图像

    将给定的向量（如PCA降维后的特征向量）按指定的图像形状展示为图片，方便进行视觉检查。

    参数：
    vectors (ndarray): 需要展示的向量，形状为(n_samples, vector_size)。
    image_shape (tuple): 每个图像的形状，默认为(64, 64)。
    num_show (int): 要展示的图像数量，默认为8。
    title (str): 图像标题，默认为空字符串。

    返回：
    None
    """
    plt.figure(figsize=(15, 8))  # 设置图像的显示尺寸
    for i in range(num_show):
        plt.subplot(2, 4, i + 1)  # 将图像显示为2行4列的子图
        plt.imshow(vectors[i].reshape(image_shape), cmap='gray')  # 将每个向量重塑为图像并显示
        plt.axis('off')  # 不显示坐标轴
    plt.suptitle(title, fontsize=20, y=0.95)  # 设置总标题，y 控制标题的垂直位置
    plt.show()  # 展示图像


def visualize_2d(X, y, title):
    """ 可视化2D数据

    使用散点图可视化2D数据，并根据标签将不同类别的点进行区分。

    参数：
    X (ndarray): 数据点的特征矩阵，形状为(n_samples, 2)，即仅考虑前两个维度。
    y (ndarray): 数据点的标签，形状为(n_samples,)，用于分类区分。
    title (str): 图像标题。

    返回：
    None
    """
    df = pd.DataFrame(X[:, :2], columns=['Dim1', 'Dim2'])  # 提取前两维数据并转换为DataFrame
    df['Label'] = y  # 将标签添加到DataFrame中
    plt.figure(figsize=(8, 6))  # 设置图像大小
    sns.scatterplot(data=df, x='Dim1', y='Dim2', hue='Label', palette='tab20', legend=False)  # 绘制散点图
    plt.title(title)  # 设置图像标题
    plt.show()  # 展示图像
