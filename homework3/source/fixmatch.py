"""
Copyright (C) 2025 傅祉珏

:module: fixmatch
:function: 实现FixMatch算法的单步训练流程
:author: 傅祉珏
:date: 2025-05-23
:license: AGPLv3 + 附加限制条款（禁止商用）

本代码依据 GNU Affero 通用公共许可证 v3 (AGPLv3) 授权，附加条款见项目根目录 ADDITIONAL_TERMS.md。
- 禁止商业用途（包括但不限于销售、集成至商业产品）
- 学术使用需在代码注释或文档中明确标注来源

AGPLv3 完整文本见 LICENSE 文件或 <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import torch
import torch.nn.functional as F


def fixmatch_step(model, optimizer, labeled_batch, unlabeled_batch_weak, unlabeled_batch_strong,
                  criterion, device, threshold=0.95, lambda_u=1):
    """ FixMatch算法的单步训练流程

    参数：
    model (nn.Module): 待训练的模型。
    optimizer (Optimizer): 优化器。
    labeled_batch (tuple): 带标签批次数据 (images, labels)。
    unlabeled_batch_weak (Tensor): 弱增强的无标签图像批次。
    unlabeled_batch_strong (Tensor): 强增强的无标签图像批次。
    criterion (nn.Module): 有监督损失函数。
    device (torch.device): 计算设备 (CPU/GPU)。
    threshold (float): 伪标签置信度阈值，默认为0.95。
    lambda_u (float): 无监督损失权重，默认为1。

    返回：
    tuple: (总损失, 有监督损失, 无监督损失, mask比例, 预测logits, 真实标签)
    """
    # 设置模型为训练模式
    model.train()

    # 解包批次数据
    x_l, y_l = labeled_batch  # 带标签数据
    x_u_w = unlabeled_batch_weak  # 弱增强无标签数据
    x_u_s = unlabeled_batch_strong  # 强增强无标签数据

    # 确保数据在正确的设备上
    batch_size = x_l.size(0)
    x_l, y_l = x_l.to(device), y_l.to(device)
    x_u_w, x_u_s = x_u_w.to(device), x_u_s.to(device)

    # 1) 计算带标签样本分类损失
    logits_x = model(x_l)  # 模型预测
    loss_x = criterion(logits_x, y_l)  # 有监督损失

    # 2) 对弱增强无标签样本预测，生成伪标签
    with torch.no_grad():  # 不计算梯度
        logits_u_w = model(x_u_w)  # 弱增强预测
        probs = F.softmax(logits_u_w, dim=1)  # 转换为概率
        max_probs, pseudo_labels = torch.max(probs, dim=1)  # 获取最大概率和对应类别
        mask = max_probs.ge(threshold).float()  # 置信度高于阈值则mask=1

    # 3) 计算无监督损失（仅对高置信度样本）
    logits_u_s = model(x_u_s)  # 强增强预测
    loss_u = (F.cross_entropy(logits_u_s, pseudo_labels, reduction='none') * mask).mean()

    # 4) 计算总损失（有监督+无监督）
    loss = loss_x + lambda_u * loss_u

    # 反向传播和参数更新
    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 反向传播
    optimizer.step()  # 参数更新

    # 返回训练指标
    return loss.item(), loss_x.item(), loss_u.item(), mask.mean().item(), logits_x, y_l
