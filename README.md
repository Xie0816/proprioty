# 地形分类项目

## 项目概述

本项目是一个基于机器学习和深度学习的车辆地形分类系统，旨在通过分析车辆传感器数据（IMU加速度、底盘速度等）自动识别车辆当前行驶的道路类型。系统支持四种道路类型的分类：土路、沥青路、水泥路和碎石路。

## 主要工作

### 1. 数据特征工程
- **传感器数据提取**：从车辆传感器数据中提取三轴加速度（X、Y、Z轴）和底盘信息
- **频域特征提取**：使用Welch方法计算加速度信号的频带功率（delta、theta、alpha、beta频带）
- **时序特征处理**：考虑时间窗口内的历史数据，提取时序模式特征

### 2. 多种机器学习模型实现
项目实现了四种不同的分类模型：

#### 聚类分类器 (Clustering Classifier)
- 使用K-means聚类提取高级特征表示
- 结合MLP神经网络进行分类

#### 多层感知机 (MLP)
- 传统的全连接神经网络
- 多层隐藏层结构

#### 支持向量机 (SVM)
- 经典的机器学习分类器
- 适用于小样本分类问题

#### 时序注意力模型 (Temporal Attention)
- 考虑时间序列依赖关系
- 使用注意力机制捕捉重要时序特征
- 专门处理时序传感器数据

### 3. 数据可视化与分析
- **t-SNE降维可视化**：将高维特征降维到2D空间进行可视化分析
- **特征分布分析**：分析不同道路类型的特征分布模式
- **模型性能评估**：详细的类别准确率分析

## 项目结构

```
proproity/
├── train_clustering.py              # 聚类分类器训练脚本
├── train_mlp.py                     # MLP模型训练脚本
├── train_svm.py                     # SVM模型训练脚本
├── train_temporal_attention.py      # 时序注意力模型训练脚本
├── tsne_visualization.py            # t-SNE可视化分析脚本
├── data/
│   ├── data_loader.py               # 数据加载器
│   ├── dataset.py                   # 基础数据集类
│   ├── temporal_dataset.py          # 时序数据集类
│   └── map_analysis.py              # 地图分析工具
├── model/
│   ├── clustering_classifier.py     # 聚类分类器模型
│   ├── mlp.py                       # MLP模型
│   ├── svm.py                       # SVM模型
│   └── temporal_attention.py        # 时序注意力模型
└── weight/
    ├── clustering/                  # 聚类模型权重和结果
    ├── mlp/                         # MLP模型权重和结果
    ├── svm/                         # SVM模型权重和结果
    └── attention/                   # 时序注意力模型权重和结果
```

## 核心代码说明

### 数据模块 (data/)

#### `data_loader.py`
- **功能**：原始数据加载器
- **主要类**：`Proprio_Loader`
- **方法**：
  - `load_roi_frame_json()`: 加载ROI帧数据
  - `load_proprio_seg_infos_json()`: 加载传感器数据

#### `dataset.py`
- **功能**：基础数据集处理
- **主要类**：`TerrainDataset`
- **特征提取**：
  - IMU加速度频域特征（12维）
  - 底盘速度信息（9维）
  - 总计21维特征向量
- **类别映射**：
  - `Dirt_Road`: 0
  - `Asphalt_Road`: 1
  - `Cement_Road`: 2
  - `Gravel_Road`: 3

#### `temporal_dataset.py`
- **功能**：时序数据集处理
- **特点**：考虑时间窗口内的连续数据
- **应用**：时序注意力模型的数据准备

### 模型模块 (model/)

#### `clustering_classifier.py`
- **架构**：K-means聚类 + MLP分类器
- **核心类**：
  - `ClusteringFeatureExtractor`: 聚类特征提取器
  - `ClusteringMLP`: 聚类MLP模型
  - `EndToEndClusteringClassifier`: 端到端分类器
- **特征维度**：原始21维 → 聚类16维 → 分类4类

#### `mlp.py`
- **架构**：多层感知机神经网络
- **隐藏层**：[128, 64, 32]神经元
- **激活函数**：ReLU
- **正则化**：Dropout (0.1)

#### `svm.py`
- **实现**：支持向量机分类器
- **核函数**：线性核
- **特点**：适用于小样本分类

#### `temporal_attention.py`
- **架构**：时序注意力机制
- **特点**：捕捉时间序列中的重要特征
- **应用**：处理传感器数据的时序依赖性

### 训练脚本

#### `train_clustering.py`
- **功能**：训练聚类分类器
- **数据集**：合并19个数据集，总计53,048个样本
- **训练配置**：
  - 训练集：42,438样本
  - 验证集：10,610样本
  - 批次大小：32
  - 学习率：0.0001
  - 训练轮次：100

#### `train_mlp.py`
- **功能**：训练MLP模型
- **架构**：21 → 128 → 64 → 32 → 4

#### `train_svm.py`
- **功能**：训练SVM模型
- **特点**：快速训练，适用于基准测试

#### `train_temporal_attention.py`
- **功能**：训练时序注意力模型
- **特点**：处理时序数据，考虑时间依赖性

### 可视化与分析

#### `tsne_visualization.py`
- **功能**：特征空间可视化
- **方法**：t-SNE降维到2D
- **输出**：可视化图表和结果文件

## 数据集

项目使用19个车辆行驶数据集，包含在不同道路条件下的传感器数据：

- **数据来源**：车辆IMU传感器和底盘控制系统
- **数据格式**：JSON格式的时间序列数据
- **特征维度**：21维特征向量
- **类别分布**：
  - 土路 (Dirt_Road): 19,191样本
  - 沥青路 (Asphalt_Road): 7,134样本
  - 水泥路 (Cement_Road): 4,208样本
  - 碎石路 (Gravel_Road): 22,515样本

## 模型性能对比

### 时序注意力模型 (最佳性能)
- **验证集准确率**：96.91%
- **训练集准确率**：95.13%
- **各类别准确率**：
  - 土路 (Dirt_Road): 96.12%
  - 沥青路 (Asphalt_Road): 98.74%
  - 水泥路 (Cement_Road): 90.94%
  - 碎石路 (Gravel_Road): 98.10%
- **验证集损失**：0.0894
- **训练集损失**：0.1251

### 聚类分类器
- **验证集准确率**：87.04%
- **训练集准确率**：85.29%
- **各类别准确率**：
  - 土路 (Dirt_Road): 84.14%
  - 沥青路 (Asphalt_Road): 92.26%
  - 水泥路 (Cement_Road): 56.65%
  - 碎石路 (Gravel_Road): 93.41%

### 多层感知机 (MLP)
- **验证集准确率**：88.40%
- **训练集准确率**：87.43%
- **各类别准确率**：
  - 土路 (Dirt_Road): 88.36%
  - 沥青路 (Asphalt_Road): 93.09%
  - 水泥路 (Cement_Road): 51.56%
  - 碎石路 (Gravel_Road): 94.53%

### 支持向量机 (SVM)
- **验证集准确率**：70.69%
- **训练集准确率**：70.49%
- **各类别准确率**：
  - 土路 (Dirt_Road): 51.83%
  - 沥青路 (Asphalt_Road): 81.06%
  - 水泥路 (Cement_Road): 0.00%
  - 碎石路 (Gravel_Road): 97.31%

### 性能总结
1. **时序注意力模型表现最佳**：96.91%的验证准确率，显著优于其他模型
2. **水泥路分类挑战**：所有模型在水泥路分类上都表现相对较差
3. **模型复杂度与性能**：时序注意力模型虽然复杂，但获得了最好的性能
4. **SVM表现最差**：在水泥路上完全无法分类，整体性能较低

## 使用说明

### 环境要求
```bash
Python 3.8+
PyTorch 1.9+
scikit-learn
numpy
matplotlib
seaborn
scipy
```

### 训练模型
```bash
# 训练聚类分类器
python train_clustering.py

# 训练MLP模型
python train_mlp.py

# 训练SVM模型
python train_svm.py

# 训练时序注意力模型
python train_temporal_attention.py
```

### 可视化分析
```bash
# 生成t-SNE可视化
python tsne_visualization.py
```
