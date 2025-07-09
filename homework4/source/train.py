"""
Copyright (C) 2025 傅祉珏 杨程骏 谢敬豪

:module: train
:function: 对比学习的训练代码框架
:author: 傅祉珏，杨程骏，谢敬豪
:date: 2025-07-04
:license: AGPLv3 + 附加限制条款（禁止商用）

本代码依据 GNU Affero 通用公共许可证 v3 (AGPLv3) 授权，附加条款见项目根目录 ADDITIONAL_TERMS.md。
- 禁止商业用途（包括但不限于销售、集成至商业产品）
- 学术使用需在代码注释或文档中明确标注来源

AGPLv3 完整文本见 LICENSE 文件或 <https://www.gnu.org/licenses/agpl-3.0.html>
"""

import time
from dataloader import *
from loss import *
from model.cart import *
from model.ensemble import *
from model.kmeans import *
from model.knn import *
from model.logistic import *
from model.mlp import *
from model.svm import *
from utils import *

# 类别名称映射
class_name = ['Distinction', 'Fail', 'Pass', 'Withdrawn']
# 模型名称列表
models_name = ['LogReg', 'SVM', 'CART', 'KNN', 'KMeans', 'MLP']

# 初始化结果存储列表
train_accs = []        # 存储各模型训练准确率
test_accs = []         # 存储各模型测试准确率
training_times = []    # 存储各模型训练时间

# 数据加载与预处理
data_loader = OULADDataLoader(data_dir='./data')
X_train, X_test, y_train, y_test = data_loader.split_data()

# 打印数据集信息
print(f"Size of Training Set: {len(X_train)}")
print(f"Size of Testing Set: {len(X_test)}")
print()

# 打印测试集真实标签分布
print("True Labels")
counts = np.bincount(y_test)
for cls, count in enumerate(counts):
    print(f"Class {cls} has {count} samples")
print()

# ==================== 逻辑回归模型 ====================
loss = cross_entropy()
lr = LogisticRegression(
    learning_rate=0.1, 
    lr_decay=0.95, 
    max_iter=100, 
    batch_size=16, 
    random_seed=3407, 
    loss=loss
)

# 训练与评估
start = time.perf_counter()
train_acc, train_loss = lr.train(X_train, y_train)
end = time.perf_counter()
test_acc, test_loss = lr.evaluate(X_test, y_test)

# 存储结果
train_accs.append(train_acc)
test_accs.append(test_acc)
training_times.append(end - start)

# 打印结果
print("========== Logistic Regression ==========")
print(f"Train Acc: {train_acc * 100:.2f}%, Train Loss: {train_loss:.4f}")
print(f"Test Acc: {test_acc * 100:.2f}%, Test Loss: {test_loss:.4f}")
print(f"Training time: {end - start:.4f} seconds")

# 绘制混淆矩阵
train_cm, test_cm = lr.confusion_matrix()
plot_confusion_matrix(train_cm, class_name, title='Training Confusion Matrix of Logistic Regression')
plot_confusion_matrix(test_cm, class_name, title='Testing Confusion Matrix of Logistic Regression')

# 打印预测分布
predict = lr.predict(X_test)
counts = np.bincount(predict)
for cls, count in enumerate(counts):
    print(f"Class {class_name[cls]} has {count} samples")
print()

# ==================== SVM模型 ====================
loss = multi_class_hinge()
svm = SVM(
    learning_rate=0.1, 
    lr_decay=0.95, 
    max_iter=100, 
    batch_size=16, 
    random_seed=3407, 
    loss=loss
)

# 训练与评估
start = time.perf_counter()
train_acc, train_loss = svm.train(X_train, y_train)
end = time.perf_counter()
test_acc, test_loss = svm.evaluate(X_test, y_test)

# 存储结果
train_accs.append(train_acc)
test_accs.append(test_acc)
training_times.append(end - start)

# 打印结果
print("========== SVM ==========")
print(f"Train Acc: {train_acc * 100:.2f}%, Train Loss: {train_loss:.4f}")
print(f"Test Acc: {test_acc * 100:.2f}%, Test Loss: {test_loss:.4f}")
print(f"Training time: {end - start:.4f} seconds")

# 绘制混淆矩阵
train_cm, test_cm = svm.confusion_matrix()
plot_confusion_matrix(train_cm, class_name, title='Training Confusion Matrix of SVM')
plot_confusion_matrix(test_cm, class_name, title='Testing Confusion Matrix of SVM')

# 打印预测分布
predict = svm.predict(X_test)
counts = np.bincount(predict)
for cls, count in enumerate(counts):
    print(f"Class {class_name[cls]} has {count} samples")
print()

# ==================== 决策树模型 ====================
loss = impurity_entropy()
cart = CART(
    learning_rate=0.1, 
    lr_decay=0.95, 
    max_iter=100, 
    batch_size=16, 
    random_seed=3407, 
    loss=loss
)

# 训练与评估
start = time.perf_counter()
train_acc, train_loss = cart.train(X_train, y_train)
end = time.perf_counter()
test_acc, test_loss = cart.evaluate(X_test, y_test)

# 存储结果
train_accs.append(train_acc)
test_accs.append(test_acc)
training_times.append(end - start)

# 打印结果
print("========== Decision Tree ==========")
print(f"Train Acc: {train_acc * 100:.2f}%, Train Loss: {train_loss:.4f}")
print(f"Test Acc: {test_acc * 100:.2f}%, Test Loss: {test_loss:.4f}")
print(f"Training time: {end - start:.4f} seconds")

# 绘制混淆矩阵
train_cm, test_cm = cart.confusion_matrix()
plot_confusion_matrix(train_cm, class_name, title='Training Confusion Matrix of Decision Tree')
plot_confusion_matrix(test_cm, class_name, title='Testing Confusion Matrix of Decision Tree')

# 打印预测分布
predict = cart.predict(X_test)
counts = np.bincount(predict)
for cls, count in enumerate(counts):
    print(f"Class {class_name[cls]} has {count} samples")
print()

# ==================== KNN模型 ====================
loss = accuracy()
knn = KNN(
    learning_rate=0.1, 
    lr_decay=0.95, 
    max_iter=100, 
    batch_size=16, 
    random_seed=3407, 
    loss=loss
)

# 训练与评估
start = time.perf_counter()
train_acc, train_loss = knn.train(X_train, y_train)
end = time.perf_counter()
test_acc, test_loss = knn.evaluate(X_test, y_test)

# 存储结果
train_accs.append(train_acc)
test_accs.append(test_acc)
training_times.append(end - start)

# 打印结果
print("========== KNN ==========")
print(f"Train Acc: {train_acc * 100:.2f}%, Train Loss: {train_loss:.4f}")
print(f"Test Acc: {test_acc * 100:.2f}%, Test Loss: {test_loss:.4f}")
print(f"Training time: {end - start:.4f} seconds")

# 绘制混淆矩阵
train_cm, test_cm = knn.confusion_matrix()
plot_confusion_matrix(train_cm, class_name, title='Training Confusion Matrix of KNN')
plot_confusion_matrix(test_cm, class_name, title='Testing Confusion Matrix of KNN')

# 打印预测分布
predict = knn.predict(X_test)
counts = np.bincount(predict)
for cls, count in enumerate(counts):
    print(f"Class {class_name[cls]} has {count} samples")
print()

# ==================== K-means模型 ====================
loss = clustering_loss()
kmeans = Kmeans(
    learning_rate=0.1, 
    lr_decay=0.95, 
    max_iter=100, 
    batch_size=16, 
    random_seed=3407, 
    loss=loss
)

# 训练与评估
start = time.perf_counter()
train_acc, train_loss = kmeans.train(X_train, y_train)
end = time.perf_counter()
test_acc, test_loss = kmeans.evaluate(X_test, y_test)

# 存储结果
train_accs.append(train_acc)
test_accs.append(test_acc)
training_times.append(end - start)

# 打印结果
print("========== K-means ==========")
print(f"Train Acc: {train_acc * 100:.2f}%, Train Loss: {train_loss:.4f}")
print(f"Test Acc: {test_acc * 100:.2f}%, Test Loss: {test_loss:.4f}")
print(f"Training time: {end - start:.4f} seconds")

# 绘制混淆矩阵
train_cm, test_cm = kmeans.confusion_matrix()
plot_confusion_matrix(train_cm, class_name, title='Training Confusion Matrix of K-means')
plot_confusion_matrix(test_cm, class_name, title='Testing Confusion Matrix of K-means')

# 打印预测分布
predict = kmeans.predict(X_test)
counts = np.bincount(predict)
for cls, count in enumerate(counts):
    print(f"Class {class_name[cls]} has {count} samples")
print()

# ==================== MLP模型 ====================
loss = cross_entropy()
mlp = MLP(
    learning_rate=0.1, 
    lr_decay=0.95, 
    max_iter=100, 
    batch_size=16, 
    random_seed=3407, 
    loss=loss
)

# 训练与评估
start = time.perf_counter()
train_acc, train_loss = mlp.train(X_train, y_train)
end = time.perf_counter()
test_acc, test_loss = mlp.evaluate(X_test, y_test)

# 存储结果
train_accs.append(train_acc)
test_accs.append(test_acc)
training_times.append(end - start)

# 打印结果
print("========== MLP ==========")
print(f"Train Acc: {train_acc * 100:.2f}%, Train Loss: {train_loss:.4f}")
print(f"Test Acc: {test_acc * 100:.2f}%, Test Loss: {test_loss:.4f}")
print(f"Training time: {end - start:.4f} seconds")

# 绘制混淆矩阵
train_cm, test_cm = mlp.confusion_matrix()
plot_confusion_matrix(train_cm, class_name, title='Training Confusion Matrix of MLP')
plot_confusion_matrix(test_cm, class_name, title='Testing Confusion Matrix of MLP')

# 打印预测分布
predict = mlp.predict(X_test)
counts = np.bincount(predict)
for cls, count in enumerate(counts):
    print(f"Class {class_name[cls]} has {count} samples")
print()

# ==================== 集成模型（逻辑回归元模型） ====================
loss = cross_entropy()
models = [lr, svm, cart, knn, kmeans, mlp]
ensemble = StackingEnsemble(
    base_models=models, 
    meta_model=LogisticRegression(
        learning_rate=0.1, 
        lr_decay=0.95, 
        max_iter=100, 
        batch_size=16, 
        random_seed=3407, 
        loss=loss
    ), 
    use_proba=True
)

# 训练与评估
start = time.perf_counter()
train_acc, train_loss = ensemble.train(X_train, y_train)
end = time.perf_counter()
test_acc, test_loss = ensemble.evaluate(X_test, y_test)

# 存储结果
train_accs.append(train_acc)
test_accs.append(test_acc)
training_times.append(end - start)

# 绘制混淆矩阵
train_cm, test_cm = ensemble.confusion_matrix()
plot_confusion_matrix(train_cm, class_name, title='Training Confusion Matrix of Ensemble (LogisticRegression)')
plot_confusion_matrix(test_cm, class_name, title='Testing Confusion Matrix of Ensemble (LogisticRegression)')

# 打印结果
print("========== Ensemble - Logistic Regression ==========")
print(f"Train Acc: {train_acc * 100:.2f}%, Train Loss: {train_loss:.4f}")
print(f"Test Acc: {test_acc * 100:.2f}%, Test Loss: {test_loss:.4f}")
print(f"Training time: {end - start:.4f} seconds")
print()

# ==================== 集成模型（SVM元模型） ====================
loss = multi_class_hinge()
models = [lr, svm, cart, knn, kmeans, mlp]
ensemble = StackingEnsemble(
    base_models=models, 
    meta_model=SVM(
        learning_rate=0.1, 
        lr_decay=0.95, 
        max_iter=100, 
        batch_size=16, 
        random_seed=3407, 
        loss=loss
    ), 
    use_proba=True
)

# 训练与评估
start = time.perf_counter()
train_acc, train_loss = ensemble.train(X_train, y_train)
end = time.perf_counter()
test_acc, test_loss = ensemble.evaluate(X_test, y_test)

# 存储结果
train_accs.append(train_acc)
test_accs.append(test_acc)
training_times.append(end - start)

# 绘制混淆矩阵
train_cm, test_cm = ensemble.confusion_matrix()
plot_confusion_matrix(train_cm, class_name, title='Training Confusion Matrix of Ensemble (SVM)')
plot_confusion_matrix(test_cm, class_name, title='Testing Confusion Matrix of Ensemble (SVM)')

# 打印结果
print("========== Ensemble - SVM ==========")
print(f"Train Acc: {train_acc * 100:.2f}%, Train Loss: {train_loss:.4f}")
print(f"Test Acc: {test_acc * 100:.2f}%, Test Loss: {test_loss:.4f}")
print(f"Training time: {end - start:.4f} seconds")
print()

# ==================== 集成模型（决策树元模型） ====================
loss = impurity_entropy()
models = [lr, svm, cart, knn, kmeans, mlp]
ensemble = StackingEnsemble(
    base_models=models, 
    meta_model=CART(
        learning_rate=0.1, 
        lr_decay=0.95, 
        max_iter=100, 
        batch_size=16, 
        random_seed=3407, 
        loss=loss
    ), 
    use_proba=True
)

# 训练与评估
start = time.perf_counter()
train_acc, train_loss = ensemble.train(X_train, y_train)
end = time.perf_counter()
test_acc, test_loss = ensemble.evaluate(X_test, y_test)

# 存储结果
train_accs.append(train_acc)
test_accs.append(test_acc)
training_times.append(end - start)

# 绘制混淆矩阵
train_cm, test_cm = ensemble.confusion_matrix()
plot_confusion_matrix(train_cm, class_name, title='Training Confusion Matrix of Ensemble (Decision Tree)')
plot_confusion_matrix(test_cm, class_name, title='Testing Confusion Matrix of Ensemble (Decision Tree)')

# 打印结果
print("========== Ensemble - Decision Tree ==========")
print(f"Train Acc: {train_acc * 100:.2f}%, Train Loss: {train_loss:.4f}")
print(f"Test Acc: {test_acc * 100:.2f}%, Test Loss: {test_loss:.4f}")
print(f"Training time: {end - start:.4f} seconds")
print()

# ==================== 集成模型（KNN元模型） ====================
loss = accuracy()
models = [lr, svm, cart, knn, kmeans, mlp]
ensemble = StackingEnsemble(
    base_models=models, 
    meta_model=KNN(
        learning_rate=0.1, 
        lr_decay=0.95, 
        max_iter=100, 
        batch_size=16, 
        random_seed=3407, 
        loss=loss
    ), 
    use_proba=True
)

# 训练与评估
start = time.perf_counter()
train_acc, train_loss = ensemble.train(X_train, y_train)
end = time.perf_counter()
test_acc, test_loss = ensemble.evaluate(X_test, y_test)

# 存储结果
train_accs.append(train_acc)
test_accs.append(test_acc)
training_times.append(end - start)

# 绘制混淆矩阵
train_cm, test_cm = ensemble.confusion_matrix()
plot_confusion_matrix(train_cm, class_name, title='Training Confusion Matrix of Ensemble (KNN)')
plot_confusion_matrix(test_cm, class_name, title='Testing Confusion Matrix of Ensemble (KNN)')

# 打印结果
print("========== Ensemble - KNN ==========")
print(f"Train Acc: {train_acc * 100:.2f}%, Train Loss: {train_loss:.4f}")
print(f"Test Acc: {test_acc * 100:.2f}%, Test Loss: {test_loss:.4f}")
print(f"Training time: {end - start:.4f} seconds")
print()

# ==================== 集成模型（Kmeans元模型） ====================
loss = clustering_loss()
models = [lr, svm, cart, knn, kmeans, mlp]
ensemble = StackingEnsemble(
    base_models=models, 
    meta_model=Kmeans(
        learning_rate=0.1, 
        lr_decay=0.95, 
        max_iter=100, 
        batch_size=16, 
        random_seed=3407, 
        loss=loss
    ), 
    use_proba=True
)

# 训练与评估
start = time.perf_counter()
train_acc, train_loss = ensemble.train(X_train, y_train)
end = time.perf_counter()
test_acc, test_loss = ensemble.evaluate(X_test, y_test)

# 存储结果
train_accs.append(train_acc)
test_accs.append(test_acc)
training_times.append(end - start)

# 绘制混淆矩阵
train_cm, test_cm = ensemble.confusion_matrix()
plot_confusion_matrix(train_cm, class_name, title='Training Confusion Matrix of Ensemble (Kmeans)')
plot_confusion_matrix(test_cm, class_name, title='Testing Confusion Matrix of Ensemble (Kmeans)')

# 打印结果
print("========== Ensemble - Kmeans ==========")
print(f"Train Acc: {train_acc * 100:.2f}%, Train Loss: {train_loss:.4f}")
print(f"Test Acc: {test_acc * 100:.2f}%, Test Loss: {test_loss:.4f}")
print(f"Training time: {end - start:.4f} seconds")
print()

# ==================== 集成模型（MLP元模型） ====================
loss = cross_entropy()
models = [lr, svm, cart, knn, kmeans, mlp]
ensemble = StackingEnsemble(
    base_models=models, 
    meta_model=MLP(
        learning_rate=0.1, 
        lr_decay=0.95, 
        max_iter=100, 
        batch_size=16, 
        random_seed=3407, 
        loss=loss
    ), 
    use_proba=True
)

# 训练与评估
start = time.perf_counter()
train_acc, train_loss = ensemble.train(X_train, y_train)
end = time.perf_counter()
test_acc, test_loss = ensemble.evaluate(X_test, y_test)

# 存储结果
train_accs.append(train_acc)
test_accs.append(test_acc)
training_times.append(end - start)

# 绘制混淆矩阵
train_cm, test_cm = ensemble.confusion_matrix()
plot_confusion_matrix(train_cm, class_name, title='Training Confusion Matrix of Ensemble (MLP)')
plot_confusion_matrix(test_cm, class_name, title='Testing Confusion Matrix of Ensemble (MLP)')

# 打印结果
print("========== Ensemble - MLP ==========")
print(f"Train Acc: {train_acc * 100:.2f}%, Train Loss: {train_loss:.4f}")
print(f"Test Acc: {test_acc * 100:.2f}%, Test Loss: {test_loss:.4f}")
print(f"Training time: {end - start:.4f} seconds")
print()

# ==================== 结果可视化 ====================
plot_accuracy(train_accs, class_names=models_name, title="Training Accuracy of Different Models")
plot_accuracy(test_accs, class_names=models_name, title="Test Accuracy of Different Models")
plot_performance(training_times, class_names=models_name, title="Performance of Different Models")
