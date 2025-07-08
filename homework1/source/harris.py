"""
Copyright (C) 2025 傅祉珏

:module: harris
:function: Harris 角点检测算法实现及测试示例
:author: 傅祉珏
:date: 2025-03-28
:license: AGPLv3 + 附加限制条款（禁止商用）

本代码依据 GNU Affero 通用公共许可证 v3 (AGPLv3) 授权，附加条款见项目根目录 ADDITIONAL_TERMS.md。
- 禁止商业用途（包括但不限于销售、集成至商业产品）
- 学术使用需在代码注释或文档中明确标注来源

AGPLv3 完整文本见 LICENSE 文件或 <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import os

import cv2 as cv
import numpy as np


def harris_corner_detect(image, k=0.04, threshold_ratio=0.001, window_size=5, non_max_supression=True):
    """ 使用 Harris 角点检测算法检测图像中的角点 """

    # 将输入图像转换为灰度图
    if len(image.shape) == 3:  # 彩色图像
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:  # 已是灰度图
        gray = image.copy()
    gray = np.float32(gray)  # 转换为浮点数类型以提高计算精度

    # 计算 x 和 y 方向的梯度
    Ix = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)  # 计算 x 方向梯度
    Iy = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)  # 计算 y 方向梯度

    # 计算梯度的平方及乘积
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy

    # 进行高斯平滑以减少噪声
    sigma = 2  # 平滑系数，值越大平滑效果越明显
    Ix2 = cv.GaussianBlur(Ix2, (window_size, window_size), sigma)
    Iy2 = cv.GaussianBlur(Iy2, (window_size, window_size), sigma)
    Ixy = cv.GaussianBlur(Ixy, (window_size, window_size), sigma)

    # 计算 Harris 角点响应值 R
    det = Ix2 * Iy2 - Ixy ** 2  # 计算结构张量的行列式
    trace = Ix2 + Iy2  # 计算结构张量的迹
    R = det - k * (trace ** 2)  # 计算响应值 R

    # 设定阈值，小于阈值的点设为 0
    threshold = threshold_ratio * np.max(R)
    R[R < threshold] = 0

    # 执行非极大值抑制以去除冗余角点
    if non_max_supression:
        kernel = np.ones((5, 5), np.uint8)  # 5x5 窗口进行最大值检测
        dilated = cv.dilate(R, kernel)  # 扩张操作获取局部最大值
        local_maxima = (R == dilated)  # 仅保留局部最大值的点
        R[~local_maxima] = 0  # 非局部最大值的点设为 0

    # 获取角点坐标
    corners = np.argwhere(R > 0)  # 找出非零位置的角点

    return corners


if __name__ == '__main__':

    # 创建用于存放检测结果的目录
    if not os.path.exists('results'):
        os.makedirs('results')

    # 读取输入图像
    image_path = 'images/sudoku.png'
    image = cv.imread(image_path)

    # 应用 Harris 角点检测
    corners = harris_corner_detect(image, k=0.04, threshold_ratio=0.001, window_size=5)

    # 在原图上绘制角点（红色圆点）
    image_out = image.copy()
    for y, x in corners:
        cv.circle(image_out, (x, y), 3, (0, 0, 255), -1)  # 半径 3 的红色圆点

    # 保存结果图像
    cv.imwrite('results/sudoku_keypoints.png', image_out)
