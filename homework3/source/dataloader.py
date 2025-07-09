"""
Copyright (C) 2025 傅祉珏

:module: dataloader
:function: 加载已划分的CIFAR-10数据集并生成训练、验证、测试数据加载器
:author: 傅祉珏
:date: 2025-05-23
:license: AGPLv3 + 附加限制条款（禁止商用）

本代码依据 GNU Affero 通用公共许可证 v3 (AGPLv3) 授权，附加条款见项目根目录 ADDITIONAL_TERMS.md。
- 禁止商业用途（包括但不限于销售、集成至商业产品）
- 学术使用需在代码注释或文档中明确标注来源

AGPLv3 完整文本见 LICENSE 文件或 <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10


class UnlabeledCIFAR10(Dataset):
    """ 无标签CIFAR-10数据集类

    属性：
    df (DataFrame): 包含图像索引的CSV文件数据。
    cifar_dataset (CIFAR10): 原始CIFAR-10数据集。
    transform_weak (Compose): 弱数据增强变换。
    transform_strong (Compose): 强数据增强变换。
    """

    def __init__(self, csv_file, cifar_dataset, transform_weak, transform_strong):
        """ 初始化无标签数据集

        参数：
        csv_file (str): 包含图像索引的CSV文件路径。
        cifar_dataset (CIFAR10): 原始CIFAR-10数据集。
        transform_weak (Compose): 弱数据增强变换。
        transform_strong (Compose): 强数据增强变换。
        """
        self.df = pd.read_csv(csv_file)
        self.cifar_dataset = cifar_dataset
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong

    def __len__(self):
        """ 获取数据集长度 """
        return len(self.df)

    def __getitem__(self, idx):
        """ 获取数据项

        参数：
        idx (int): 数据索引。

        返回：
        tuple: (弱增强图像, 强增强图像)
        """
        img_index = int(self.df.iloc[idx]['index'])
        image, _ = self.cifar_dataset[img_index]
        return self.transform_weak(image), self.transform_strong(image)


class CIFAR10FromCSV(Dataset):
    """ 从CSV加载的CIFAR-10数据集类

    属性：
    df (DataFrame): 包含图像索引和标签的CSV文件数据。
    cifar_dataset (CIFAR10): 原始CIFAR-10数据集。
    transform (Compose): 数据增强变换。
    """

    def __init__(self, csv_file, cifar_dataset, transform=None):
        """ 初始化数据集

        参数：
        csv_file (str): 包含图像索引和标签的CSV文件路径。
        cifar_dataset (CIFAR10): 原始CIFAR-10数据集。
        transform (Compose): 数据增强变换，默认为None。
        """
        self.df = pd.read_csv(csv_file)
        self.cifar_dataset = cifar_dataset
        self.transform = transform

    def __len__(self):
        """ 获取数据集长度 """
        return len(self.df)

    def __getitem__(self, idx):
        """ 获取数据项

        参数：
        idx (int): 数据索引。

        返回：
        tuple: (图像, 标签)
        """
        img_index = int(self.df.iloc[idx]['index'])
        label = int(self.df.iloc[idx]['label'])
        image, _ = self.cifar_dataset[img_index]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_cifar10_transforms(image_size=32):
    """ 获取CIFAR-10数据增强变换

    参数：
    image_size (int): 图像尺寸，默认为32。

    返回：
    tuple: (训练集变换, 测试集变换)
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomCrop(image_size, padding=4),  # 随机裁剪
        transforms.ToTensor(),  # 转为张量
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],  # 标准化
                             std=[0.247, 0.243, 0.261]),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.247, 0.243, 0.261]),
    ])
    return train_transform, test_transform


def get_cifar10_dataloaders(train_csv, val_csv, test_csv, root="./datasets/data",
                            batch_size=64, num_workers=4, pin_memory=True):
    """ 获取CIFAR-10数据加载器

    参数：
    train_csv (str): 训练集CSV文件路径。
    val_csv (str): 验证集CSV文件路径。
    test_csv (str): 测试集CSV文件路径。
    root (str): 数据集存储路径，默认为"./datasets/data"。
    batch_size (int): 批大小，默认为64。
    num_workers (int): 数据加载工作线程数，默认为4。
    pin_memory (bool): 是否使用固定内存，默认为True。

    返回：
    tuple: (训练集加载器, 验证集加载器, 测试集加载器, 类别列表)
    """
    # 加载完整数据集
    train_set_full = CIFAR10(root=root, train=True, download=True)
    test_set_full = CIFAR10(root=root, train=False, download=True)

    # 获取数据增强
    train_transform, test_transform = get_cifar10_transforms()

    # 创建数据集
    train_dataset = CIFAR10FromCSV(train_csv, train_set_full, transform=train_transform)
    val_dataset = CIFAR10FromCSV(val_csv, train_set_full, transform=test_transform)
    test_dataset = CIFAR10FromCSV(test_csv, test_set_full, transform=test_transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader, list(range(10))


def get_ssl_dataloaders(labeled_csv, unlabeled_csv, val_csv, test_csv,
                        root="./datasets/data", batch_size=64, num_workers=4, pin_memory=True):
    """ 获取半监督学习数据加载器

    参数：
    labeled_csv (str): 有标签训练集CSV文件路径。
    unlabeled_csv (str): 无标签训练集CSV文件路径。
    val_csv (str): 验证集CSV文件路径。
    test_csv (str): 测试集CSV文件路径。
    root (str): 数据集存储路径，默认为"./datasets/data"。
    batch_size (int): 批大小，默认为64。
    num_workers (int): 数据加载工作线程数，默认为4。
    pin_memory (bool): 是否使用固定内存，默认为True。

    返回：
    tuple: (有标签加载器, 无标签加载器, 验证集加载器, 测试集加载器)
    """
    # 加载完整数据集
    train_set_full = CIFAR10(root=root, train=True, download=True)
    test_set = CIFAR10(root=root, train=False, download=True)

    # 定义标准化参数
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.247, 0.243, 0.261])

    # 定义弱增强变换
    transform_weak = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        normalize
    ])

    # 定义强增强变换
    transform_strong = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandAugment(),  # 随机增强
        transforms.ToTensor(),
        normalize
    ])

    # 定义测试集变换
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # 创建数据集
    labeled_dataset = CIFAR10FromCSV(labeled_csv, train_set_full, transform=transform_weak)
    unlabeled_dataset = UnlabeledCIFAR10(unlabeled_csv, train_set_full,
                                         transform_weak, transform_strong)
    val_dataset = CIFAR10FromCSV(val_csv, train_set_full, transform=test_transform)
    test_dataset = CIFAR10FromCSV(test_csv, test_set, transform=test_transform)

    # 创建数据加载器
    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=pin_memory)

    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    return labeled_loader, unlabeled_loader, val_loader, test_loader
