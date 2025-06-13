"""
@module mixmatch
@function 实现MixMatch算法的单步训练流程
@author 傅祉珏
@date 2025年5月23日
"""

import torch
import torch.nn.functional as F
import numpy as np


def sharpen(p, T=0.5):
    """ 锐化概率分布

    参数：
    p (Tensor): 输入概率分布。
    T (float): 温度参数，控制锐化程度，默认为0.5。

    返回：
    Tensor: 锐化后的概率分布。
    """
    p_power = p ** (1 / T)  # 温度缩放
    return p_power / p_power.sum(dim=1, keepdim=True)  # 重新归一化


def mixup(x1, y1, x2, y2, alpha=0.75):
    """ MixUp数据增强

    参数：
    x1 (Tensor): 第一批次输入数据。
    y1 (Tensor): 第一批次目标。
    x2 (Tensor): 第二批次输入数据。
    y2 (Tensor): 第二批次目标。
    alpha (float): Beta分布参数，控制混合比例，默认为0.75。

    返回：
    tuple: (混合后的输入, 混合后的目标)
    """
    lam = np.random.beta(alpha, alpha)  # 从Beta分布采样混合系数
    lam = max(lam, 1 - lam)  # 保证lam >= 0.5，增强混合效果
    x = lam * x1 + (1 - lam) * x2  # 输入混合
    y = lam * y1 + (1 - lam) * y2  # 目标混合
    return x, y


def mixmatch_step(model, optimizer, labeled_batch, unlabeled_batch, criterion, device, K=2, T=0.5, alpha=0.75,
                  lambda_u=75):
    """ MixMatch算法的单步训练流程

    参数：
    model (nn.Module): 待训练的模型。
    optimizer (Optimizer): 优化器。
    labeled_batch (tuple): 带标签批次数据 (images, labels)。
    unlabeled_batch (tuple): 无标签批次数据 (weak_aug, strong_aug)。
    criterion (nn.Module): 有监督损失函数。
    device (torch.device): 计算设备 (CPU/GPU)。
    K (int): 无标签数据预测次数，默认为2。
    T (float): 锐化温度参数，默认为0.5。
    alpha (float): MixUp参数，默认为0.75。
    lambda_u (float): 无监督损失权重，默认为75。

    返回：
    tuple: (总损失, 有监督损失, 无监督损失, 预测logits, 混合目标)
    """
    # 设置模型为训练模式
    model.train()

    # 解包数据
    x_l, y_l = labeled_batch  # 有标签数据
    x_u_w, x_u_s = unlabeled_batch  # 无标签数据（弱增强和强增强）

    # 转移数据到设备
    x_l, y_l = x_l.to(device), y_l.to(device)
    x_u_w, x_u_s = x_u_w.to(device), x_u_s.to(device)

    batch_size = x_l.size(0)  # 有标签样本数

    # 1) 生成无标签数据的伪标签
    with torch.no_grad():  # 不计算梯度
        preds = []
        for _ in range(K):  # K次预测取平均
            pred = F.softmax(model(x_u_w), dim=1)  # 弱增强预测
            preds.append(pred)

        # 平均预测概率并锐化
        p_hat = torch.stack(preds, dim=0).mean(dim=0)  # 平均预测
        p_hat = sharpen(p_hat, T)  # 锐化概率分布

    # 2) 准备MixUp的输入
    # 合并有标签和无标签数据
    all_inputs = torch.cat([x_l, x_u_s], dim=0)  # 输入数据
    all_targets = torch.cat([
        F.one_hot(y_l, num_classes=10).float(),  # 有标签数据的one-hot编码
        p_hat  # 无标签数据的锐化伪标签
    ], dim=0)  # 目标数据

    # 3) 打乱顺序进行MixUp
    idx = torch.randperm(all_inputs.size(0))  # 随机排列索引
    shuffled_input = all_inputs[idx]  # 打乱输入
    shuffled_target = all_targets[idx]  # 打乱目标

    # 执行MixUp
    mixed_input, mixed_target = mixup(
        all_inputs, all_targets,
        shuffled_input, shuffled_target,
        alpha
    )

    # 4) 前向传播
    logits = model(mixed_input)  # 混合数据预测

    # 5) 计算损失
    logits_x = logits[:batch_size]  # 有标签部分预测
    logits_u = logits[batch_size:]  # 无标签部分预测

    # 有监督损失（交叉熵）
    loss_x = criterion(logits_x, y_l)

    # 无监督损失（预测与混合目标的MSE）
    loss_u = F.mse_loss(F.softmax(logits_u, dim=1), mixed_target[batch_size:])

    # 总损失（加权和）
    loss = loss_x + lambda_u * loss_u

    # 反向传播和参数更新
    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 反向传播
    optimizer.step()  # 参数更新

    return loss.item(), loss_x.item(), loss_u.item(), logits, mixed_target
