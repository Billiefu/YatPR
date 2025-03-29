"""
@module registration
@function 基于 sift 及 ransac 完成图像配准任务
@author 傅祉珏
@date 2025年3月29日
"""

import copy

import cv2 as cv
import numpy as np
from tqdm import trange

import ransac
import sift
import utils


def stitch_images(image1, image2, H, save_path, alpha=0.2):
    """ 图像拼接并叠加热力图显示对准效果 """

    # 确保图像为三通道 (兼容灰度图)
    if len(image1.shape) == 2:
        image1 = cv.cvtColor(image1, cv.COLOR_GRAY2BGR)
    if len(image2.shape) == 2:
        image2 = cv.cvtColor(image2, cv.COLOR_GRAY2BGR)

    # 生成 image1 的热力图
    gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    heatmap = cv.applyColorMap(gray1, cv.COLORMAP_JET)

    # 计算变换参数
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # 计算变换后 image1 的角点
    corners = np.array([[0, 0], [w1, 0], [0, h1], [w1, h1], [0, 0]], dtype=np.float32)
    transformed_corners = cv.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2)

    # 计算新画布尺寸
    x_min, y_min = np.floor(transformed_corners.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(transformed_corners.max(axis=0)).astype(int)
    new_width = max(x_max, w2) - min(x_min, 0)
    new_height = max(y_max, h2) - min(y_min, 0)

    # 创建平移矩阵
    shift_x = -x_min if x_min < 0 else 0
    shift_y = -y_min if y_min < 0 else 0
    shift_matrix = np.array([[1, 0, shift_x], [0, 1, shift_y], [0, 0, 1]])

    # 更新变换矩阵
    H_shifted = shift_matrix @ H

    # 变换热力图和原图
    warped_heatmap = cv.warpPerspective(heatmap, H_shifted, (new_width, new_height))
    warped_image1 = cv.warpPerspective(image1, H_shifted, (new_width, new_height))

    # 创建画布并放置 image2
    canvas = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    canvas[shift_y:shift_y + h2, shift_x:shift_x + w2] = image2

    # 叠加半透明热力图
    mask = (warped_heatmap.sum(axis=2) > 0)  # 创建非零区域掩膜
    canvas[mask] = cv.addWeighted(warped_heatmap[mask], alpha, canvas[mask], 1 - alpha, 0)

    # 叠加原始图像边界 (增强可视化)
    edges = cv.Canny(warped_image1, 100, 200)
    canvas[edges > 0] = [0, 255, 0]  # 用绿色标记边界

    # 保存结果
    cv.imwrite(save_path, canvas)


if __name__ == '__main__':

    # 分别对6张图像进行处理
    for i in trange(6):
        image1 = cv.imread(f"./images/mr_T1/T1-{i+1}.png")
        image2 = cv.imread(f"./images/mr_T2/T2-{i+1}.png")

        # 使用opencv提供的方法计算 SIFT 特征子
        keypoints1, descriptor1 = sift.sift_descriptor_opencv(cv.cvtColor(copy.deepcopy(image1), cv.COLOR_RGB2GRAY))
        keypoints2, descriptor2 = sift.sift_descriptor_opencv(cv.cvtColor(copy.deepcopy(image2), cv.COLOR_RGB2GRAY))
        matches_sift = utils.match_features(descriptor1, descriptor2)
        utils.draw_matches(image1, image2, keypoints1, keypoints2, matches_sift, f"./results/mri/match_opencv_{i+1}.png")
        if len(matches_sift) >= 4:
            H_sift = ransac.ransac_homography(keypoints1, keypoints2, matches_sift)
            stitch_images(image1, image2, H_sift, f"./results/mri/compare_opencv_{i+1}.png")

        # 使用自己编写的优化方法计算 SIFT 特征子
        descriptor1 = sift.sift_descriptor_optimal(cv.cvtColor(copy.deepcopy(image1), cv.COLOR_RGB2GRAY), keypoints1)
        descriptor2 = sift.sift_descriptor_optimal(cv.cvtColor(copy.deepcopy(image2), cv.COLOR_RGB2GRAY), keypoints2)
        matches_sift = utils.match_features(descriptor1, descriptor2)
        utils.draw_matches(image1, image2, keypoints1, keypoints2, matches_sift, f"./results/mri/match_handon_{i+1}.png")
        if len(matches_sift) >= 4:
            H_sift = ransac.ransac_homography(keypoints1, keypoints2, matches_sift)
            stitch_images(image1, image2, H_sift, f"./results/mri/compare_handon_{i+1}.png")
