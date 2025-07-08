"""
Copyright (C) 2025 傅祉珏

:module: utils
:function: 存放所有可能需要用到的共性函数
:author: 傅祉珏
:date: 2025-03-28
:license: AGPLv3 + 附加限制条款（禁止商用）

本代码依据 GNU Affero 通用公共许可证 v3 (AGPLv3) 授权，附加条款见项目根目录 ADDITIONAL_TERMS.md。
- 禁止商业用途（包括但不限于销售、集成至商业产品）
- 学术使用需在代码注释或文档中明确标注来源

AGPLv3 完整文本见 LICENSE 文件或 <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import cv2 as cv
import numpy as np


def gaussian_kernel(size, sigma):
    """ 生成一个高斯滤波器的卷积核 """

    k = size // 2  # 计算高斯核的半径
    y, x = np.mgrid[-k:k + 1, -k:k + 1]  # 创建网格坐标
    g = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))  # 计算高斯函数
    return g / g.sum()  # 对高斯核进行归一化


def gaussian_blur(img, size=5, sigma=1.6):
    """ 使用高斯核对输入图像进行模糊处理，减少噪声 """

    kernel = gaussian_kernel(size, sigma)  # 生成高斯核
    return cv.filter2D(img, -1, kernel)  # 对图像进行卷积操作，执行高斯模糊


def compute_dog_pyramid(img, num_intervals=3, sigma=1.6):
    """ 生成高斯金字塔并计算相邻两层之间的差分图像（DoG）用于后续的特征检测 """

    k = 2 ** (1 / num_intervals)  # 计算每一层的标准差增长因子
    gaussian_images = [gaussian_blur(img, sigma=sigma * (k ** i)) for i in range(num_intervals + 3)]  # 生成高斯金字塔
    dog_pyramid = [gaussian_images[i + 1] - gaussian_images[i] for i in range(len(gaussian_images) - 1)]  # 计算差分图像
    return dog_pyramid  # 返回DoG金字塔


def detect_keypoints(dog_pyramid):
    """ 在差分金字塔（DoG）空间中查找局部极值点，作为图像的关键点 """

    keypoints = []  # 用于存储检测到的关键点
    for i in range(1, len(dog_pyramid) - 1):
        prev, curr, next_ = dog_pyramid[i - 1], dog_pyramid[i], dog_pyramid[i + 1]  # 获取相邻的三层
        for y in range(1, curr.shape[0] - 1):  # 遍历当前图层的每个像素
            for x in range(1, curr.shape[1] - 1):
                # 提取三层的3x3邻域（确保为单通道）
                patch_prev = prev[y - 1:y + 2, x - 1:x + 2]
                patch_curr = curr[y - 1:y + 2, x - 1:x + 2]
                patch_next = next_[y - 1:y + 2, x - 1:x + 2]
                # 合并为3x3x3的立方体
                patch = np.stack([patch_prev, patch_curr, patch_next], axis=0)
                # 当前点值（标量）
                current_val = curr[y, x]
                # 检查极值
                if np.abs(current_val) > 0.03:  # 只检测显著的点
                    if current_val == np.max(patch) or current_val == np.min(patch):
                        keypoints.append((x, y))  # 如果是极值，则添加到关键点列表
    return keypoints  # 返回所有检测到的关键点


def match_features(descriptor1, descriptor2):
    """ 利用FLANN算法加速计算两个描述符之间的最近邻匹配 """

    FLANN_INDEX_KDTREE = 1  # 使用KD树作为FLANN的搜索方式
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # 设置FLANN索引参数
    search_params = dict(checks=50)  # 设置搜索参数，控制精度与速度的平衡

    flann = cv.FlannBasedMatcher(index_params, search_params)  # 创建FLANN匹配器
    matches = flann.knnMatch(descriptor1, descriptor2, k=2)  # 使用k近邻匹配

    # 应用 Lowe's 比率测试，筛选出良好的匹配
    good_matches = [m[0] for m in matches if m[0].distance < 0.7 * m[1].distance]  # 比率测试
    return good_matches  # 返回良好的匹配


def draw_matches(image1, image2, keypoints1, keypoints2, matches, output_path):
    """ 绘制两幅图像之间的匹配结果，并将其保存为输出文件 """

    # 检查每个关键点是否为 cv.KeyPoint 类型
    if not all(isinstance(keypoint, cv.KeyPoint) for keypoint in keypoints1):
        keypoints1 = [cv.KeyPoint(x=kp[0], y=kp[1], size=1) for kp in keypoints1]  # 转换为cv.KeyPoint对象

    if not all(isinstance(keypoint, cv.KeyPoint) for keypoint in keypoints2):
        keypoints2 = [cv.KeyPoint(x=kp[0], y=kp[1], size=1) for kp in keypoints2]  # 转换为cv.KeyPoint对象

    # 绘制前50对匹配
    match_img = cv.drawMatches(
        image1, keypoints1,
        image2, keypoints2,
        matches[:50], None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,  # 不绘制未匹配的点
    )
    cv.imwrite(output_path, match_img)  # 保存匹配结果图像
