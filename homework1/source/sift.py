"""
Copyright (C) 2025 傅祉珏

:module: sift
:function: 存放获取 SIFT 特征描述子的方法
:author: 傅祉珏
:date: 2025-03-28
:license: AGPLv3 + 附加限制条款（禁止商用）

本代码依据 GNU Affero 通用公共许可证 v3 (AGPLv3) 授权，附加条款见项目根目录 ADDITIONAL_TERMS.md。
- 禁止商业用途（包括但不限于销售、集成至商业产品）
- 学术使用需在代码注释或文档中明确标注来源

AGPLv3 完整文本见 LICENSE 文件或 <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import copy
import os

import cv2 as cv
import numpy as np

import harris
import ransac
import utils


def sift_descriptor_handon(image, keypoints, window_size=16):
    """ 计算 SIFT 特征描述子（手工实现版本） """

    descriptors = []
    for keypoint in keypoints:
        # 提取关键点坐标（支持 cv.KeyPoint 和元组格式）
        if isinstance(keypoint, cv.KeyPoint):
            x = int(keypoint.pt[0])
            y = int(keypoint.pt[1])
        else:
            x, y = int(keypoint[0]), int(keypoint[1])  # 假设是元组或列表格式

        # 边界检查，避免超出图像范围
        if (x < window_size // 2 or x >= image.shape[1] - window_size // 2 or
                y < window_size // 2 or y >= image.shape[0] - window_size // 2):
            continue

        # 截取局部窗口
        y1, y2 = y - window_size // 2, y + window_size // 2
        x1, x2 = x - window_size // 2, x + window_size // 2
        patch = image[y1:y2, x1:x2]

        # 计算梯度幅值和方向
        grad_x = cv.Sobel(patch, cv.CV_64F, 1, 0, ksize=3)
        grad_y = cv.Sobel(patch, cv.CV_64F, 0, 1, ksize=3)
        magnitude = np.hypot(grad_x, grad_y)
        orientation = np.degrees(np.arctan2(grad_y, grad_x)) % 360

        # 构建 SIFT 描述子（4x4 子块，每个子块 8 个方向）
        descriptor = np.zeros(128, dtype=np.float32)
        cell_size = window_size // 4  # 每个子块大小（4x4）

        for i in range(4):
            for j in range(4):
                # 提取子块区域
                y_start = i * cell_size
                y_end = (i + 1) * cell_size
                x_start = j * cell_size
                x_end = (j + 1) * cell_size

                sub_mag = magnitude[y_start:y_end, x_start:x_end]
                sub_ori = orientation[y_start:y_end, x_start:x_end]

                # 计算方向直方图（8 个 bins）
                hist, _ = np.histogram(sub_ori, bins=8, range=(0, 360), weights=sub_mag)
                descriptor[i * 32 + j * 8: i * 32 + j * 8 + 8] = hist

        # 归一化
        descriptor /= np.linalg.norm(descriptor) + 1e-6
        descriptors.append(descriptor)

    return np.array(descriptors)


def sift_descriptor_optimal(image, keypoints, window_size=16):
    """ 优化的 SIFT 特征描述子生成（高斯加权 + 三线性插值） """

    # 1. 进行镜像填充，处理边界关键点
    pad = window_size // 2
    image_padded = cv.copyMakeBorder(image, pad, pad, pad, pad, cv.BORDER_REFLECT_101)

    # 2. 预计算高斯加权窗口
    sigma = window_size / 3.0
    x = np.arange(-pad, pad)
    gaussian_1d = np.exp(-x ** 2 / (2 * sigma ** 2))
    gaussian_2d = np.outer(gaussian_1d, gaussian_1d)
    gaussian_2d /= gaussian_2d.sum()  # 归一化

    descriptors = []

    for keypoint in keypoints:
        # 3. 获取关键点坐标，并转换为填充图像中的坐标
        if isinstance(keypoint, cv.KeyPoint):
            x_orig, y_orig = keypoint.pt
        else:
            x_orig, y_orig = keypoint[0], keypoint[1]

        x = int(x_orig) + pad
        y = int(y_orig) + pad

        # 4. 提取局部窗口
        patch = image_padded[y - pad:y + pad, x - pad:x + pad]

        # 5. 计算梯度（使用 Scharr 算子提高精度）
        grad_x = cv.Scharr(patch, cv.CV_32F, 1, 0)
        grad_y = cv.Scharr(patch, cv.CV_32F, 0, 1)

        # 6. 计算带高斯加权的梯度幅值和方向
        magnitude = np.hypot(grad_x, grad_y) * gaussian_2d
        orientation = np.degrees(np.arctan2(grad_y, grad_x)) % 360

        # 7. 初始化描述子
        descriptor = np.zeros(128, dtype=np.float32)
        cell_size = window_size // 4

        # 8. 进行方向和空间插值
        for dy in range(window_size):
            for dx in range(window_size):
                mag = magnitude[dy, dx]
                angle = orientation[dy, dx]

                # 方向插值
                bin_center = angle / (360 / 8)
                bin_left = int(np.floor(bin_center)) % 8
                bin_right = (bin_left + 1) % 8
                right_weight = bin_center - bin_left
                left_weight = 1 - right_weight

                # 空间插值
                y_cell = dy / cell_size - 0.5
                x_cell = dx / cell_size - 0.5
                i0 = int(np.floor(y_cell))
                j0 = int(np.floor(x_cell))

                for di in [0, 1]:
                    for dj in [0, 1]:
                        if 0 <= i0 + di < 4 and 0 <= j0 + dj < 4:
                            y_weight = 1 - abs(y_cell - (i0 + di))
                            x_weight = 1 - abs(x_cell - (j0 + dj))
                            spatial_weight = y_weight * x_weight

                            idx = ((i0 + di) * 4 + (j0 + dj)) * 8
                            descriptor[idx + bin_left] += mag * left_weight * spatial_weight
                            descriptor[idx + bin_right] += mag * right_weight * spatial_weight

        # 9. 归一化和截断
        norm = np.linalg.norm(descriptor) + 1e-6
        descriptor /= norm
        descriptor = np.clip(descriptor, 0, 0.2)
        renorm = np.linalg.norm(descriptor) + 1e-6
        descriptor /= renorm

        descriptors.append(descriptor)

    return np.array(descriptors)


def sift_descriptor_opencv(image):
    """ 使用 OpenCV 计算 SIFT 关键点和特征描述子 """

    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


if __name__ == '__main__':

    # 创建存放结果的目录
    if not os.path.exists('results'):
        os.makedirs('results')

    # 读取图像
    image_path1 = "images/uttower1.jpg"
    image_path2 = "images/uttower2.jpg"
    image1 = cv.imread(image_path1)
    image2 = cv.imread(image_path2)

    # 提取 Harris 角点
    corners = harris.harris_corner_detect(image1)
    image1_keypoints = copy.deepcopy(image1)
    for y, x in corners:
        cv.circle(image1_keypoints, (x, y), 3, (0, 0, 255), -1)  # 红色标记角点
    cv.imwrite('results/' + os.path.basename(image_path1).replace(".jpg", "_keypoints.jpg"), image1_keypoints)

    # 使用opencv提供的方法计算 SIFT 特征子
    keypoints1, descriptor1 = sift_descriptor_opencv(cv.cvtColor(copy.deepcopy(image1), cv.COLOR_RGB2GRAY))
    keypoints2, descriptor2 = sift_descriptor_opencv(cv.cvtColor(copy.deepcopy(image2), cv.COLOR_RGB2GRAY))
    matches_sift = utils.match_features(descriptor1, descriptor2)
    utils.draw_matches(image1, image2, keypoints1, keypoints2, matches_sift, "results/opencv/uttower_match_sift.png")
    H_sift = ransac.ransac_homography(keypoints1, keypoints2, matches_sift)
    ransac.stitch_images(image1, image2, H_sift, "results/opencv/uttower_stitching_sift.png")

    # 使用自己编写的方法计算 SIFT 特征子
    descriptor1 = sift_descriptor_handon(cv.cvtColor(copy.deepcopy(image1), cv.COLOR_RGB2GRAY), keypoints1)
    descriptor2 = sift_descriptor_handon(cv.cvtColor(copy.deepcopy(image2), cv.COLOR_RGB2GRAY), keypoints2)
    matches_sift = utils.match_features(descriptor1, descriptor2)
    utils.draw_matches(image1, image2, keypoints1, keypoints2, matches_sift, "results/handon/uttower_match_sift.png")
    H_sift = ransac.ransac_homography(keypoints1, keypoints2, matches_sift)
    ransac.stitch_images(image1, image2, H_sift, "results/handon/uttower_stitching_sift.png")

    # 使用自己编写的优化方法计算 SIFT 特征子
    descriptor1 = sift_descriptor_optimal(cv.cvtColor(copy.deepcopy(image1), cv.COLOR_RGB2GRAY), keypoints1)
    descriptor2 = sift_descriptor_optimal(cv.cvtColor(copy.deepcopy(image2), cv.COLOR_RGB2GRAY), keypoints2)
    matches_sift = utils.match_features(descriptor1, descriptor2)
    utils.draw_matches(image1, image2, keypoints1, keypoints2, matches_sift, "results/handon/uttower_match_sift_opt.png")
    H_sift = ransac.ransac_homography(keypoints1, keypoints2, matches_sift)
    ransac.stitch_images(image1, image2, H_sift, "results/handon/uttower_stitching_sift_opt.png")
