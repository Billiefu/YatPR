"""
@module hog
@function 存放获取 HOG 特征描述子的方法
@author 傅祉珏
@date 2025年3月28日
"""

import copy
import os

import cv2 as cv
import numpy as np

import harris
import ransac
import utils


def hog_factor_handon(image, keypoints, cell_size=8, bins=9, win_size=(64, 64)):
    """ 改进的 HOG 特征计算（支持关键点归一化处理） """

    # 1. 调整图像尺寸，使其为 cell_size 的整数倍
    h, w = image.shape
    h = (h // cell_size) * cell_size
    w = (w // cell_size) * cell_size
    image = cv.resize(image, (w, h))

    # 2. 预处理：高斯模糊 + 梯度计算
    image = cv.GaussianBlur(image, (5, 5), 1.5)
    grad_x = cv.Scharr(image, cv.CV_32F, 1, 0)
    grad_y = cv.Scharr(image, cv.CV_32F, 0, 1)

    # 计算梯度幅值和方向（角度范围为 0 到 180 度）
    magnitude = np.hypot(grad_x, grad_y)
    orientation = np.degrees(np.arctan2(grad_y, grad_x)) % 180

    # 3. 计算每个 Cell 的梯度方向直方图
    cell_h, cell_w = h // cell_size, w // cell_size
    hog_descriptor = np.zeros((cell_h, cell_w, bins))

    for y in range(cell_h):
        for x in range(cell_w):
            # 获取当前 Cell 的梯度幅值和方向
            cell_mag = magnitude[y * cell_size:(y + 1) * cell_size, x * cell_size:(x + 1) * cell_size]
            cell_ori = orientation[y * cell_size:(y + 1) * cell_size, x * cell_size:(x + 1) * cell_size]

            # 计算方向直方图（9 个 bins）
            hist = np.zeros(bins)
            for dy in range(cell_size):
                for dx in range(cell_size):
                    angle = cell_ori[dy, dx]
                    mag = cell_mag[dy, dx]

                    # 进行双线性插值，将梯度值分配到相邻的两个方向 bin
                    bin_idx = angle / (180 / bins)
                    left_bin = int(np.floor(bin_idx)) % bins
                    right_bin = (left_bin + 1) % bins
                    right_weight = bin_idx - left_bin
                    left_weight = 1 - right_weight

                    hist[left_bin] += mag * left_weight
                    hist[right_bin] += mag * right_weight
            hog_descriptor[y, x, :] = hist

    # 4. 关键点归一化（基于 HOG 直方图的局部块特征）
    descriptors = []
    half_win = win_size[0] // 2 // cell_size  # 计算窗口的一半大小（单位：cell）

    for keypoint in keypoints:
        # 转换关键点坐标为 Cell 坐标
        x_cell = int(keypoint[0] / cell_size)
        y_cell = int(keypoint[1] / cell_size)

        # 处理边界情况（镜像填充）
        pad_left = max(0, half_win - x_cell)
        pad_right = max(0, (x_cell + half_win) - cell_w)
        pad_top = max(0, half_win - y_cell)
        pad_bottom = max(0, (y_cell + half_win) - cell_h)

        # 进行镜像填充
        padded_hog = np.pad(hog_descriptor,
                            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                            mode='reflect')

        # 计算新坐标
        new_x = x_cell + pad_left
        new_y = y_cell + pad_top

        # 提取局部块并进行归一化
        block = padded_hog[new_y - half_win:new_y + half_win,
                new_x - half_win:new_x + half_win, :]
        block_flat = block.reshape(-1)

        # L2-Hys 归一化（先归一化，截断后再归一化）
        norm = np.linalg.norm(block_flat) + 1e-6
        block_norm = block_flat / norm
        block_norm = np.clip(block_norm, 0, 0.2)
        renorm = np.linalg.norm(block_norm) + 1e-6
        descriptors.append(block_norm / renorm)

    return np.array(descriptors, dtype=np.float32)


def hog_factor_cv(image, keypoints):
    """ 使用 OpenCV 计算 HOG 关键点特征 """

    # 设定 HOG 参数
    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    descriptors = []

    for keypoint in keypoints:
        # 计算关键点窗口
        x, y = int(keypoint[0]), int(keypoint[1])
        x1, y1 = max(0, x - 32), max(0, y - 32)
        x2, y2 = min(image.shape[1], x + 32), min(image.shape[0], y + 32)

        # 提取窗口并计算 HOG 特征
        patch = cv.resize(image[y1:y2, x1:x2], win_size)
        descriptors.append(hog.compute(patch).flatten())

    return np.array(descriptors)


if __name__ == '__main__':

    # 创建保存目录
    if not os.path.exists('results'):
        os.makedirs('results')

    # 读取两幅图像
    image_path1 = "images/uttower1.jpg"
    image_path2 = "images/uttower2.jpg"
    image1 = cv.imread(image_path1)
    image2 = cv.imread(image_path2)

    # 使用 Harris 角点检测
    corners = harris.harris_corner_detect(image1)
    image1_keypoints = copy.deepcopy(image1)
    for y, x in corners:
        cv.circle(image1_keypoints, (x, y), 3, (0, 0, 255), -1)  # 以红色标记角点
    cv.imwrite('results/' + os.path.basename(image_path1).replace(".jpg", "_keypoints.jpg"), image1_keypoints)

    # 找出两幅图像的关键点
    dog_pyramid1 = utils.compute_dog_pyramid(cv.cvtColor(copy.deepcopy(image1), cv.COLOR_BGR2GRAY))
    dog_pyramid2 = utils.compute_dog_pyramid(cv.cvtColor(copy.deepcopy(image2), cv.COLOR_BGR2GRAY))
    keypoints1 = utils.detect_keypoints(dog_pyramid1)
    keypoints2 = utils.detect_keypoints(dog_pyramid2)

    # 使用自己编写的方法计算 HOG 特征子
    descriptor1_hog = hog_factor_handon(cv.cvtColor(copy.deepcopy(image1), cv.COLOR_RGB2GRAY), keypoints1)
    descriptor2_hog = hog_factor_handon(cv.cvtColor(copy.deepcopy(image2), cv.COLOR_RGB2GRAY), keypoints2)
    matches_hog_handon = utils.match_features(descriptor1_hog, descriptor2_hog)
    utils.draw_matches(image1, image2, keypoints1, keypoints2, matches_hog_handon, "results/handon/uttower_match_hog.png")
    H_hog = ransac.ransac_homography(keypoints1, keypoints2, matches_hog_handon)
    ransac.stitch_images(image1, image2, H_hog, "results/handon/uttower_stitching_hog.png")

    # 使用 Opencv 提供的方法计算 HOG 特征子
    descriptor1_hog = hog_factor_cv(cv.cvtColor(copy.deepcopy(image1), cv.COLOR_RGB2GRAY), keypoints1)
    descriptor2_hog = hog_factor_cv(cv.cvtColor(copy.deepcopy(image2), cv.COLOR_RGB2GRAY), keypoints2)
    matches_hog = utils.match_features(descriptor1_hog, descriptor2_hog)
    utils.draw_matches(image1, image2, keypoints1, keypoints2, matches_hog, "results/opencv/uttower_match_hog.png")
    H_hog = ransac.ransac_homography(keypoints1, keypoints2, matches_hog)
    ransac.stitch_images(image1, image2, H_hog, "results/opencv/uttower_stitching_hog.png")
