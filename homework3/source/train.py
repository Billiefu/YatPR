"""
@module train
@function 半监督学习模型训练主程序，支持FixMatch和MixMatch算法
@author 傅祉珏
@date 2025年5月23日
"""

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 解决libiomp5md.dll冲突问题

import gc
import random

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

from dataloader import get_ssl_dataloaders
from fixmatch import fixmatch_step
from mixmatch import mixmatch_step
from models.wideresnet import WideResNet
from utils import set_seed, plot_curves, plot_loss

# =================== 配置参数 =====================
method = "fixmatch"  # 训练方法选择："mixmatch"或"fixmatch"
batch_size = 64  # 批次大小
num_epochs = 1024  # 训练总轮数
learning_rate = 0.002  # 学习率
lambda_u = 75 if method == "mixmatch" else 1  # 无监督损失权重
accumulation_steps = 1  # 梯度累积步数
val_check_interval = 100  # 验证间隔
patience = 10  # 早停耐心值
save_path = f"./models/{method}.pth"  # 模型保存路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备选择

# =================== 数据路径 =====================
train_csv = "datasets/cifar10_train_4000.csv"  # 训练集CSV路径
val_csv = "datasets/cifar10_val_4000.csv"  # 验证集CSV路径
test_csv = "datasets/cifar10_test.csv"  # 测试集CSV路径

# =================== 超参数 =====================
num_labeled = 250  # 使用的有标签样本数量

# =================== 初始化设置 =====================
set_seed(42)  # 设置随机种子
torch.backends.cudnn.benchmark = True  # 启用CuDNN基准测试


def split_labeled_unlabeled(train_csv, num_labeled, seed=42):
    """ 划分有标签和无标签数据集

    参数：
    train_csv (str): 原始训练集CSV路径
    num_labeled (int): 需要的有标签样本数量
    seed (int): 随机种子，默认为42

    返回：
    tuple: (有标签数据集路径, 无标签数据集路径)
    """
    df = pd.read_csv(train_csv)
    random.seed(seed)

    labeled_indices = []
    n_per_class = num_labeled // 10  # 每类样本数

    # 按类别均衡采样
    for c in range(10):
        cls_indices = df[df['label'] == c].index.tolist()
        sampled = random.sample(cls_indices, n_per_class)
        labeled_indices.extend(sampled)

    unlabeled_indices = list(set(df.index) - set(labeled_indices))

    # 创建临时CSV文件
    labeled_df = df.loc[labeled_indices].reset_index(drop=True)
    unlabeled_df = df.loc[unlabeled_indices].reset_index(drop=True)

    labeled_path = "datasets/temp_labeled.csv"
    unlabeled_path = "datasets/temp_unlabeled.csv"
    labeled_df.to_csv(labeled_path, index=False)
    unlabeled_df.to_csv(unlabeled_path, index=False)

    return labeled_path, unlabeled_path


def main():
    """ 主训练函数 """
    # 1. 数据准备
    labeled_csv, unlabeled_csv = split_labeled_unlabeled(train_csv, num_labeled=num_labeled)

    # 获取数据加载器
    labeled_loader, unlabeled_loader, val_loader, test_loader = get_ssl_dataloaders(
        labeled_csv, unlabeled_csv, val_csv, test_csv, batch_size=batch_size
    )

    # 2. 模型初始化
    model = WideResNet(depth=28, widen_factor=2, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 3. 训练状态初始化
    best_val_acc, best_model = 0.0, None
    no_improve_epochs = 0
    train_loss_list, test_loss_list, train_acc_list, test_acc_list = [], [], [], []

    # 4. 训练循环
    for epoch in trange(num_epochs):
        model.train()
        total_loss, total_x, total_u = 0.0, 0.0, 0.0
        correct, total = 0, 0

        # 批次训练
        for batch_idx, (labeled_batch, unlabeled_batch) in enumerate(zip(labeled_loader, unlabeled_loader)):
            try:
                with torch.amp.autocast(enabled=True, device_type='cuda'):
                    # 根据方法选择训练步骤
                    if method == "mixmatch":
                        loss, loss_x, loss_u, preds, targets = mixmatch_step(
                            model, optimizer, labeled_batch,
                            unlabeled_batch, criterion, device, lambda_u=lambda_u
                        )
                    elif method == "fixmatch":
                        loss, loss_x, loss_u, mask_rate, preds, targets = fixmatch_step(
                            model, optimizer, labeled_batch,
                            unlabeled_batch[0], unlabeled_batch[1],
                            criterion, device, lambda_u=lambda_u
                        )
                    else:
                        raise ValueError(f"Unsupported method: {method}")

                # 统计损失
                total_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
                total_x += loss_x.item() if isinstance(loss_x, torch.Tensor) else loss_x
                total_u += loss_u.item() if isinstance(loss_u, torch.Tensor) else loss_u

                # 计算准确率
                _, y_l = labeled_batch
                y_l = y_l.to(device)
                pred_labels = preds.argmax(dim=1)
                correct += (pred_labels[:len(y_l)] == y_l).sum().item()
                total += len(y_l)

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"❌ CUDA OOM at batch {batch_idx}, skipping...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e

        # 5. 记录训练指标
        avg_loss = total_loss / min(len(labeled_loader), len(unlabeled_loader))
        train_loss_list.append(avg_loss)
        train_acc = 100. * correct / total
        train_acc_list.append(train_acc)

        # 6. 验证阶段
        val_acc, val_loss = evaluate(model, val_loader, criterion, device)
        print(f"✅ Epoch {epoch + 1}: Train Loss {avg_loss:.4f} | Train Acc {train_acc:.2f}% | Val Acc {val_acc:.2f}%")

        # 7. 早停机制
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print("⏹️ Early stopping triggered.")
                break

        # 8. 定期测试
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            test_acc, test_loss = evaluate(model, test_loader, criterion, device)
            test_acc_list.append(test_acc)
            test_loss_list.append(test_loss)
            print(f"📊 Test Accuracy: {test_acc:.2f}%, Test Loss: {test_loss:.4f}")

        # 清理缓存
        torch.cuda.empty_cache()
        gc.collect()

    # 9. 保存与可视化
    if best_model:
        model.load_state_dict(best_model)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Best model saved to {save_path}")

    plot_curves(train_acc_list, test_acc_list, save_path=f"result/{method}_acc_curve.png")
    plot_loss(train_loss_list, test_loss_list, save_path=f"result/{method}_loss_curve.png")


def evaluate(model, dataloader, criterion, device):
    """ 模型评估函数

    参数：
    model (nn.Module): 待评估模型
    dataloader (DataLoader): 数据加载器
    criterion (nn.Module): 损失函数
    device (torch.device): 计算设备

    返回：
    tuple: (准确率, 平均损失)
    """
    model.eval()
    correct, total = 0, 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    return acc, avg_loss


if __name__ == "__main__":
    main()
