"""
@module wideresnet
@function 实现Wide Residual Network (WideResNet)模型
@author 傅祉珏
@date 2025年5月23日
"""

import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """ 基础残差块模块

    属性：
    bn1 (BatchNorm2d): 第一个批归一化层。
    relu1 (ReLU): 第一个ReLU激活函数。
    conv1 (Conv2d): 第一个卷积层。
    bn2 (BatchNorm2d): 第二个批归一化层。
    relu2 (ReLU): 第二个ReLU激活函数。
    conv2 (Conv2d): 第二个卷积层。
    drop_rate (float): dropout概率。
    equal_in_out (bool): 输入输出维度是否相同标志。
    shortcut (nn.Module): 捷径连接层。
    """

    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        """ 初始化基础残差块

        参数：
        in_planes (int): 输入通道数。
        out_planes (int): 输出通道数。
        stride (int): 卷积步长。
        drop_rate (float): dropout概率，默认为0.0。
        """
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False)
        self.drop_rate = drop_rate
        self.equal_in_out = in_planes == out_planes
        self.shortcut = (lambda x: x) if self.equal_in_out else nn.Conv2d(in_planes, out_planes, 1, stride, bias=False)

    def forward(self, x):
        """ 前向传播

        参数：
        x (Tensor): 输入张量。

        返回：
        Tensor: 输出张量。
        """
        out = self.relu1(self.bn1(x))
        shortcut = self.shortcut(x)
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return out + shortcut


class NetworkBlock(nn.Module):
    """ 网络块模块，由多个基础残差块组成

    属性：
    layer (Sequential): 基础残差块序列。
    """

    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        """ 初始化网络块

        参数：
        nb_layers (int): 基础残差块数量。
        in_planes (int): 输入通道数。
        out_planes (int): 输出通道数。
        block (nn.Module): 基础残差块类型。
        stride (int): 第一个块的卷积步长。
        drop_rate (float): dropout概率，默认为0.0。
        """
        super().__init__()
        layers = [block(in_planes if i == 0 else out_planes, out_planes, stride if i == 0 else 1, drop_rate)
                  for i in range(nb_layers)]
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        """ 前向传播

        参数：
        x (Tensor): 输入张量。

        返回：
        Tensor: 输出张量。
        """
        return self.layer(x)


class WideResNet(nn.Module):
    """ WideResNet主网络

    属性：
    conv1 (Conv2d): 初始卷积层。
    block1-3 (NetworkBlock): 三个网络块。
    bn1 (BatchNorm2d): 最终批归一化层。
    relu (ReLU): 最终ReLU激活函数。
    fc (Linear): 全连接分类层。
    """

    def __init__(self, depth, widen_factor, num_classes, drop_rate=0.0):
        """ 初始化WideResNet

        参数：
        depth (int): 网络深度。
        widen_factor (int): 宽度扩展因子。
        num_classes (int): 分类类别数。
        drop_rate (float): dropout概率，默认为0.0。
        """
        super().__init__()
        n = (depth - 4) // 6  # 每个阶段的块数
        k = widen_factor  # 宽度因子
        nStages = [16, 16 * k, 32 * k, 64 * k]  # 各阶段通道数

        # 网络结构定义
        self.conv1 = nn.Conv2d(3, nStages[0], 3, 1, 1, bias=False)
        self.block1 = NetworkBlock(n, nStages[0], nStages[1], BasicBlock, 1, drop_rate)
        self.block2 = NetworkBlock(n, nStages[1], nStages[2], BasicBlock, 2, drop_rate)
        self.block3 = NetworkBlock(n, nStages[2], nStages[3], BasicBlock, 2, drop_rate)
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nStages[3], num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x):
        """ 前向传播

        参数：
        x (Tensor): 输入图像张量。

        返回：
        Tensor: 分类输出。
        """
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)  # 全局平均池化
        out = out.view(out.size(0), -1)
        return self.fc(out)
