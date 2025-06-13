"""
@module dataset
@function 用于将CIFAR-10数据集划分为训练集、验证集及测试集
@author 傅祉珏
@date 2025年5月23日
"""

import os
import random

import pandas as pd
from torchvision.datasets import CIFAR10


def make_cifar10_csv(root="./data", save_dir="./", num_labeled=250, val_ratio=0.1, seed=42):
    """ 生成CIFAR-10数据集的CSV划分文件

    参数：
    root (str): 数据集存储根目录，默认为"./data"。
    save_dir (str): CSV文件保存目录，默认为当前目录。
    num_labeled (int): 标记样本总数，默认为250。
    val_ratio (float): 验证集比例，默认为0.1。
    seed (int): 随机种子，默认为42。
    """
    random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    # 自动下载数据集
    print("📥 检查并加载 CIFAR-10 数据集...")
    train_dataset = CIFAR10(root=root, train=True, download=True)
    test_dataset = CIFAR10(root=root, train=False, download=True)

    train_data = train_dataset.data  # 训练集图像数据 (ndarray)
    train_targets = train_dataset.targets  # 训练集标签 (list)
    num_classes = 10  # CIFAR-10类别数

    # 每类样本数（确保类均衡）
    n_per_class = num_labeled // num_classes
    labeled_indices = []  # 存储所有标记样本的索引

    # 为每个类别选择样本
    for c in range(num_classes):
        cls_indices = [i for i, label in enumerate(train_targets) if label == c]
        selected = random.sample(cls_indices, n_per_class)
        labeled_indices.extend(selected)

    # 从 labeled_indices 中划分验证集
    val_size = int(val_ratio * len(labeled_indices))
    val_indices = random.sample(labeled_indices, val_size)
    train_indices = list(set(labeled_indices) - set(val_indices))

    # 生成CSV数据框
    train_df = pd.DataFrame({
        "index": train_indices,
        "label": [train_targets[i] for i in train_indices]
    })
    val_df = pd.DataFrame({
        "index": val_indices,
        "label": [train_targets[i] for i in val_indices]
    })
    test_df = pd.DataFrame({
        "index": list(range(len(test_dataset))),
        "label": test_dataset.targets
    })

    # 定义CSV文件路径
    train_path = os.path.join(save_dir, f"cifar10_train_{num_labeled}.csv")
    val_path = os.path.join(save_dir, f"cifar10_val_{num_labeled}.csv")
    test_path = os.path.join(save_dir, "cifar10_test.csv")

    # 保存CSV文件
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"✅ Saved train csv to {train_path} ({len(train_df)} samples)")
    print(f"✅ Saved val csv to {val_path} ({len(val_df)} samples)")
    print(f"✅ Saved test csv to {test_path} ({len(test_df)} samples)")


if __name__ == "__main__":
    # 为不同数量的标记样本生成CSV文件
    for num_labeled in [40, 250, 4000]:
        make_cifar10_csv(num_labeled=num_labeled)
