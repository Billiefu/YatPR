"""
Copyright (C) 2025 傅祉珏 谢敬豪

:module: utils
:function: 存放项目通用工具函数，包括随机种子设置、可视化等
:author: 傅祉珏，谢敬豪
:date: 2025-07-04
:license: AGPLv3 + 附加限制条款（禁止商用）

本代码依据 GNU Affero 通用公共许可证 v3 (AGPLv3) 授权，附加条款见项目根目录 ADDITIONAL_TERMS.md。
- 禁止商业用途（包括但不限于销售、集成至商业产品）
- 学术使用需在代码注释或文档中明确标注来源

AGPLv3 完整文本见 LICENSE 文件或 <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(cm, class_names=None, title="Confusion Matrix"):
    """绘制混淆矩阵热力图

    参数：
    cm (ndarray): 混淆矩阵数据，shape=(n_classes, n_classes)
    class_names (list): 类别名称列表，默认为None
    title (str): 图表标题，默认为"Confusion Matrix"

    返回：
    无返回值，直接显示图表
    """
    plt.figure(figsize=(7, 5))
    # 使用seaborn绘制热力图
    sns.heatmap(
        cm,
        annot=True,  # 显示数值
        fmt='d',  # 整数格式
        cmap='Blues',  # 蓝色调色板
        xticklabels=class_names if class_names else "auto",
        yticklabels=class_names if class_names else "auto"
    )
    plt.title(title)
    plt.xlabel("Predicted Label")  # x轴标签
    plt.ylabel("True Label")  # y轴标签
    plt.tight_layout()  # 自动调整子图参数
    plt.show()


def plot_performance(times, class_names=None, title="Performance"):
    """绘制模型性能对比图（训练时间）

    参数：
    times (list): 各模型训练时间列表
    class_names (list): 模型名称列表，默认为None
    title (str): 图表标题，默认为"Performance"

    返回：
    无返回值，直接显示图表
    """
    # 数据拆分：前6个为普通模型，后6个为集成模型
    normal_performances = times[:6]
    ensemble_performances = times[6:]
    x = np.arange(len(class_names))  # x轴坐标

    # 创建图形
    plt.figure(figsize=(7, 5))
    # 绘制普通模型性能曲线
    plt.plot(x, normal_performances, marker='o', color='red', label='Normal')
    # 绘制集成模型性能曲线
    plt.plot(x, ensemble_performances, marker='s', color='green', label='Ensemble')

    # 添加数值标签
    for i, (normal, ensemble) in enumerate(zip(normal_performances, ensemble_performances)):
        plt.text(i, normal, f"{normal:.2f}", ha='center', va='bottom', color='red')
        plt.text(i, ensemble, f"{ensemble:.2f}", ha='center', va='bottom', color='green')

    # 设置图表属性
    plt.xticks(x, class_names)  # 设置x轴刻度
    plt.xlabel('Model')  # x轴标签
    plt.ylabel('Time (s)')  # y轴标签
    plt.title(title)  # 图表标题
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格
    plt.tight_layout()  # 自动调整子图参数
    plt.show()


def plot_accuracy(accs, class_names=None, title="Accuracy"):
    """绘制模型准确率对比柱状图

    参数：
    accs (list): 各模型准确率列表
    class_names (list): 模型名称列表，默认为None
    title (str): 图表标题，默认为"Accuracy"

    返回：
    无返回值，直接显示图表
    """
    # 数据拆分：前6个为普通模型，后6个为集成模型
    normal_accs = accs[:6]
    ensemble_accs = accs[6:]

    # 柱状图参数设置
    bar_width = 0.35  # 柱状图宽度
    x = np.arange(len(class_names))  # x轴坐标

    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(7, 5))

    # 绘制普通模型准确率柱状图
    ax.bar(x - bar_width / 2, normal_accs, width=bar_width, color='skyblue', label='Normal')
    # 绘制集成模型准确率柱状图
    ax.bar(x + bar_width / 2, ensemble_accs, width=bar_width, color='orange', label='Ensemble')

    # 设置图表属性
    ax.set_xlabel('Model Type')  # x轴标签
    ax.set_ylabel('Accuracy (%)')  # y轴标签
    ax.set_title(title)  # 图表标题
    ax.set_xticks(x)  # 设置x轴刻度
    ax.set_xticklabels(class_names)  # 设置x轴刻度标签
    ax.legend()  # 显示图例

    # 添加数值标签
    for i in range(len(class_names)):
        ax.text(x[i] - bar_width / 2, normal_accs[i] + 0.5,
                f"{normal_accs[i]:.1f}%", ha='center', va='bottom', fontsize=8)
        ax.text(x[i] + bar_width / 2, ensemble_accs[i] + 0.5,
                f"{ensemble_accs[i]:.1f}%", ha='center', va='bottom', fontsize=8)

    plt.ylim(40, 90)  # 设置y轴范围
    plt.tight_layout()  # 自动调整子图参数
    plt.show()
