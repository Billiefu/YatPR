"""
Copyright (C) 2025 傅祉珏

:module: yosemite
:function: 拼接yosemite图像
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

import ransac
import sift
import utils

# 创建保存拼接结果的目录，如果目录不存在则创建
if not os.path.exists('results'):
    os.makedirs('results')

# 加载四张yosemite图像
images = [cv.imread('images/yosemite1.jpg'),
          cv.imread('images/yosemite2.jpg'),
          cv.imread('images/yosemite3.jpg'),
          cv.imread('images/yosemite4.jpg')]

# 为三张基准图像创建深拷贝，后续在这些图像上进行拼接
base_image1 = copy.deepcopy(images[0])
base_image2 = copy.deepcopy(images[0])
base_image3 = copy.deepcopy(images[0])

# 对于剩下的每一张图像，依次与基准图像进行拼接
for next_image in images[1:]:

    # 使用opencv提供的方法计算 SIFT 特征子
    keypoints1, descriptor1 = sift.sift_descriptor_opencv(cv.cvtColor(copy.deepcopy(base_image1), cv.COLOR_RGB2GRAY))
    keypoints2, descriptor2 = sift.sift_descriptor_opencv(cv.cvtColor(copy.deepcopy(next_image), cv.COLOR_RGB2GRAY))
    matches_sift = utils.match_features(descriptor1, descriptor2)
    H = ransac.ransac_homography(keypoints1, keypoints2, matches_sift)
    ransac.stitch_images(base_image1, next_image, H, "results/opencv/yosemite_stitching.png")
    base_image1 = cv.imread('results/opencv/yosemite_stitching.png')

    # 使用自己编写的方法计算 SIFT 特征子
    descriptor1 = sift.sift_descriptor_handon(cv.cvtColor(copy.deepcopy(base_image2), cv.COLOR_RGB2GRAY), keypoints1)
    descriptor2 = sift.sift_descriptor_handon(cv.cvtColor(copy.deepcopy(next_image), cv.COLOR_RGB2GRAY), keypoints2)
    matches_sift = utils.match_features(descriptor1, descriptor2)
    H = ransac.ransac_homography(keypoints1, keypoints2, matches_sift)
    ransac.stitch_images(base_image2, next_image, H, "results/handon/yosemite_stitching.png")
    base_image2 = cv.imread('results/handon/yosemite_stitching.png')

    # 使用自己编写的优化方法计算 SIFT 特征子
    descriptor1 = sift.sift_descriptor_optimal(cv.cvtColor(copy.deepcopy(base_image3), cv.COLOR_RGB2GRAY), keypoints1)
    descriptor2 = sift.sift_descriptor_optimal(cv.cvtColor(copy.deepcopy(next_image), cv.COLOR_RGB2GRAY), keypoints2)
    matches_sift = utils.match_features(descriptor1, descriptor2)
    H = ransac.ransac_homography(keypoints1, keypoints2, matches_sift)
    ransac.stitch_images(base_image3, next_image, H, "results/handon/yosemite_stitching_opt.png")
    base_image3 = cv.imread('results/handon/yosemite_stitching_opt.png')
