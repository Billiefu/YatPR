## Homework1 全景图拼接

### 实验目的

1. 熟悉 Harris 角点检测器的原理和基本使用
2. 熟悉 RANSAC 抽样一致方法的使用场景
3. 熟悉 HOG 描述子的基本原理

### 实验要求

1. 提交实验报告，要求有适当步骤说明和结果分析，对比
2. 将代码和结果打包提交
3. 实验可以使用现有的特征描述子实现

### 实验内容

1. 使用 Harris 焦点检测器寻找关键点。
2. 构建描述算子来描述图中的每个关键点，比较两幅图像的两组描述子，并进行匹配。
3. 根据一组匹配关键点，使用 RANSAC 进行仿射变换矩阵的计算。
4. 将第二幅图变换过来并覆盖在第一幅图上，拼接形成一个全景图像。
5. 实现不同的描述子，并得到不同的拼接结果。

### 实验过程

#### Harris 角点算法

请实现 **Harris 角点检测算法**，并简单阐述相关原理，对 `images/` 目录下的 `sudoku.png` 图像进行角点检测（适当进行后处理），输出对应的角点检测结果，保存到 `results/` 目录下，命名为 `sudoku\_keypoints.png`。

#### 关键点描述与匹配

请使用实现的 Harris 角点检测算法提取 `images/uttower1.jpg` 和 `images/uttower2.jpg` 的关键点，并将提取的关键点检测结果保存到 `results/`目录下，命名为 `uttower1\_keypoints.jpg`和 `uttower2\_keypoints.jpg`。

分别使用 SIFT 特征和 HOG 特征作为描述子获得两幅图像的关键点的特征，使用欧几里得距离作为特征之间相似度的度量，并绘制两幅图像之间的关键点匹配的情况，将匹配结果保存到 `results/` 目录下，命名为 `uttower\_match\_sift.png` 和 `uttower\_match\_hog.png`。使用RANSAC求解仿射变换矩阵，实现图像的拼接，并将最后拼接的结果保存到 `results/`目录下，命名为`uttower\_stitching\_sift.png`和`uttower\_stitching\_hog.png`。并分析对比 SIFT 特征和 HOG 特征在关键点匹配过程中的差异。

请将基于 SIFT+RANSAC 的拼接方法用到多张图像上，对 `images/yosemite1.png`，`images/yosemite2.png`， `images/yosemite3.png`，`images/yosemite4.png` 进行拼接，并将结果保存到 `results/` 目录下，命名为 `yosemite\_stitching.png`。

拓展：HOG 相关内容参考：[【特征检测】HOG特征算法\_比较两幅图像相似性-基于hog特征-CSDN博客](https://blog.csdn.net/hujingshuang/article/details/47337707)
