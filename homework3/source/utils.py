"""
@module utils
@function 存放项目通用工具函数，包括随机种子设置、可视化等
@author 傅祉珏
@date 2025年5月23日
"""

import random
import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed=42):
    """ 设置所有随机种子以保证实验可复现性

    参数：
    seed (int): 随机种子值，默认为42

    说明：
    设置PyTorch、NumPy、Python内置random的随机种子，
    并配置CuDNN为确定性模式以消除GPU计算的随机性
    """
    torch.manual_seed(seed)  # 设置PyTorch CPU随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU随机种子
    np.random.seed(seed)  # 设置NumPy随机种子
    random.seed(seed)  # 设置Python内置random随机种子

    # 配置CuDNN
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法相同
    torch.backends.cudnn.benchmark = False  # 禁用基准优化，保证可复现性


def plot_curves(train_acc, test_acc, save_path="acc_curve.png"):
    """ 绘制训练和测试准确率曲线

    参数：
    train_acc (list): 训练准确率列表
    test_acc (list): 测试准确率列表
    save_path (str): 图像保存路径，默认为"acc_curve.png"

    返回：
    无返回值，直接保存图像到指定路径
    """
    plt.figure(figsize=(10, 6))  # 创建新图形
    plt.plot(train_acc, label='Train Accuracy', linewidth=2)  # 绘制训练曲线
    plt.plot(test_acc, label='Test Accuracy', linewidth=2)  # 绘制测试曲线
    plt.xlabel('Epoch', fontsize=12)  # x轴标签
    plt.ylabel('Accuracy (%)', fontsize=12)  # y轴标签
    plt.title('Model Accuracy Curve', fontsize=14)  # 标题
    plt.legend(fontsize=12)  # 图例
    plt.grid(True, linestyle='--', alpha=0.7)  # 网格线
    plt.tight_layout()  # 自动调整子图参数
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 保存图像
    plt.close()  # 关闭图形


def plot_loss(train_loss, test_loss, save_path="loss_curve.png"):
    """ 绘制训练和测试损失曲线

    参数：
    train_loss (list): 训练损失列表
    test_loss (list): 测试损失列表
    save_path (str): 图像保存路径，默认为"loss_curve.png"

    返回：
    无返回值，直接保存图像到指定路径
    """
    plt.figure(figsize=(10, 6))  # 创建新图形
    plt.plot(train_loss, label='Train Loss', linewidth=2)  # 绘制训练曲线
    plt.plot(test_loss, label='Test Loss', linewidth=2)  # 绘制测试曲线
    plt.xlabel('Epoch', fontsize=12)  # x轴标签
    plt.ylabel('Loss Value', fontsize=12)  # y轴标签
    plt.title('Model Loss Curve', fontsize=14)  # 标题
    plt.legend(fontsize=12)  # 图例
    plt.grid(True, linestyle='--', alpha=0.7)  # 网格线
    plt.tight_layout()  # 自动调整子图参数
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 保存图像
    plt.close()  # 关闭图形
