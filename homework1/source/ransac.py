"""
@module ransac
@function 存放基于 RANSAC 算法的矩阵计算方法及图像拼接方法
@author 傅祉珏
@date 2025年3月28日
"""

import cv2 as cv
import numpy as np


def ransac_homography(keypoints1, keypoints2, matches):
    """ RANSAC 计算单应性矩阵（修复坐标访问） """

    # 提取源点和目标点坐标
    if all(isinstance(kp, cv.KeyPoint) for kp in keypoints1):
        # 如果 keypoints1 是 KeyPoint 对象，提取其坐标
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    else:
        # 如果 keypoints1 是元组列表，直接使用元组中的坐标
        src_pts = np.float32([keypoints1[m.queryIdx] for m in matches]).reshape(-1, 1, 2)

    if all(isinstance(kp, cv.KeyPoint) for kp in keypoints2):
        # 如果 keypoints2 是 KeyPoint 对象，提取其坐标
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    else:
        # 如果 keypoints2 是元组列表，直接使用元组中的坐标
        dst_pts = np.float32([keypoints2[m.trainIdx] for m in matches]).reshape(-1, 1, 2)

    # 使用 RANSAC 算法计算单应性矩阵 H
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    return H  # 返回计算得到的单应性矩阵


def stitch_images(image1, image2, H, save_path):
    """ 利用变换矩阵进行图像拼接 """

    # 获取 image1 的尺寸
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # 计算 image1 在新坐标系下的 4 个角点
    corners_image1 = np.array([
        [0, 0, 1], [w1, 0, 1], [0, h1, 1], [w1, h1, 1]
    ]).T  # 3x4 矩阵 (x, y, 1)

    # 变换 image1 的角点
    transformed_corners_image1 = H @ corners_image1
    transformed_corners_image1 /= transformed_corners_image1[2]  # 归一化，转换回二维坐标

    # 计算新的边界范围
    x_min = min(transformed_corners_image1[0])  # 计算变换后图像的最小 x 坐标
    y_min = min(transformed_corners_image1[1])  # 计算变换后图像的最小 y 坐标
    x_max = max(transformed_corners_image1[0])  # 计算变换后图像的最大 x 坐标
    y_max = max(transformed_corners_image1[1])  # 计算变换后图像的最大 y 坐标

    # 计算平移矩阵，防止图像变换后超出范围
    shift_x = -x_min if x_min < 0 else 0  # 如果图像有负偏移，平移图像
    shift_y = -y_min if y_min < 0 else 0  # 如果图像有负偏移，平移图像
    shift_matrix = np.array([
        [1, 0, shift_x],
        [0, 1, shift_y],
        [0, 0, 1]
    ])

    # 更新 H 矩阵，使 image1 变换后处于正确位置
    H_shifted = shift_matrix @ H

    # 计算拼接后的新尺寸
    new_width = int(max(x_max + shift_x, w2 + shift_x))  # 计算拼接后的宽度
    new_height = int(max(y_max + shift_y, h2 + shift_y))  # 计算拼接后的高度

    # 使用变换矩阵 H_shifted 对 image1 进行透视变换
    result = cv.warpPerspective(image1, H_shifted, (new_width, new_height))

    # 确保 image2 不会覆盖 image1
    x_offset = int(shift_x)  # 计算偏移量
    y_offset = int(shift_y)

    # 将 image2 放到正确的位置
    result[y_offset:y_offset + h2, x_offset:x_offset + w2] = image2

    # 保存拼接结果图像
    cv.imwrite(save_path, result)  # 将拼接结果保存到指定路径
