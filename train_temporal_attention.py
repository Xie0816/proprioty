import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import numpy as np
import random
from data.temporal_dataset import TemporalTerrainDataset
from model.temporal_attention import create_temporal_attention_model


class CombinedTemporalTerrainDataset(Dataset):
    """
    合并多个TemporalTerrainDataset的数据集类
    """
    def __init__(self, dataset_paths, time_window_seconds=1.0, sampling_rate=50):
        self.datasets = []
        self.samples = []
        
        for dataset_path in dataset_paths:
            try:
                dataset = TemporalTerrainDataset(
                    dataset_path, 
                    time_window_seconds=time_window_seconds,
                    sampling_rate=sampling_rate
                )
                self.datasets.append(dataset)
                # 合并所有样本
                self.samples.extend(dataset.samples)
                print(f"成功加载时序数据集: {dataset_path}, 样本数: {len(dataset)}")
            except Exception as e:
                print(f"加载时序数据集 {dataset_path} 失败: {e}")
        
        if len(self.samples) == 0:
            raise ValueError("没有成功加载任何时序数据集")
        
        # 统计标签分布
        self.label_counts = {}
        for sample in self.samples:
            label = sample['label']
            if label not in self.label_counts:
                self.label_counts[label] = 0
            self.label_counts[label] += 1
        
        # 打乱所有样本顺序
        random.shuffle(self.samples)
        print(f"总共加载 {len(self.datasets)} 个时序数据集，总样本数: {len(self.samples)}")
        
        # 打印标签统计信息
        self.print_label_statistics()
    
    def print_label_statistics(self):
        """打印数据集的标签统计信息"""
        print("\n=== 时序数据集标签统计 ===")
        if not self.datasets:
            print("没有可用的类别映射信息")
            return
        
        class_mapping = self.datasets[0].classes
        total_samples = len(self.samples)
        
        print(f"总样本数: {total_samples}")
        print("各类别样本分布:")
        
        for label_idx, count in self.label_counts.items():
            class_name = class_mapping.get(label_idx, f"未知类别_{label_idx}")
            percentage = (count / total_samples) * 100
            print(f"  - {class_name} (标签 {label_idx}): {count} 个样本 ({percentage:.2f}%)")
        
        # 打印时序特征维度
        if len(self.samples) > 0:
            sample = self.samples[0]
            temporal_features = sample['temporal_features']
            n_features, time_steps = temporal_features.shape
            print(f"时序特征维度: {n_features} × {time_steps}")
        
        print("===========================\n")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # 时序特征形状为 (n_features, time_steps)
        temporal_features = torch.tensor(sample['temporal_features'], dtype=torch.float32)
        label = torch.tensor(sample['label'], dtype=torch.long)
        return temporal_features, label
    
    @property
    def classes(self):
        """返回类别映射（使用第一个数据集的类别映射）"""
        if self.datasets:
            return self.datasets[0].classes
        return {}
    
    def get_feature_dimensions(self):
        """获取特征维度信息"""
        if len(self.samples) == 0:
            return None, None
        
        sample = self.samples[0]
        temporal_features = sample['temporal_features']
        n_features, time_steps = temporal_features.shape
        
        return n_features, time_steps


# 定义所有数据集路径（统一配置）
DATASET_PATHS = [
    "./dataset/2024-09-22-16-12-28",
    "./dataset/2024-09-22-16-40-20",
    "./dataset/2024-09-26-15-35-26",
    "./dataset/2025-02-25-11-43-52",
    "./dataset/2025-02-25-11-45-15",
    "./dataset/2025-02-25-11-45-55",
    "./dataset/2025-02-25-11-47-21",
    "./dataset/2025-02-25-11-47-45",
    "./dataset/2025-02-25-14-26-20",
    "./dataset/2025-02-25-14-27-13",
    "./dataset/2025-02-25-10-54-50",
    "./dataset/2025-02-25-10-55-52",
    "./dataset/2025-02-25-10-57-01",
    "./dataset/2025-02-25-10-58-08",
    "./dataset/2025-02-25-11-03-50",
    "./dataset/2025-02-25-11-04-48",
    "./dataset/2025-02-25-11-08-24",
    "./dataset/2025-02-25-11-08-54",
    "./dataset/2025-02-25-11-21-54"
]


def train_temporal_attention_model():
    """
    训练时序注意力分类模型（全量数据，包含训练集和验证集）
    """
    print("开始训练时序注意力分类模型（全量数据）...")
    
    # 时序参数
    time_window_seconds = 1.0
    sampling_rate = 50
    
    # 1. 创建合并时序数据集
    print("创建合并时序数据集...")
    try:
        full_dataset = CombinedTemporalTerrainDataset(
            DATASET_PATHS,
            time_window_seconds=time_window_seconds,
            sampling_rate=sampling_rate
        )
    except Exception as e:
        print(f"创建时序数据集失败: {e}")
        return None, None, None, None
    
    if len(full_dataset) == 0:
        print("没有有效的时序训练样本")
        return None, None, None, None
    
    # 2. 分割数据集为训练集和验证集 (80% 训练, 20% 验证)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"时序数据集分割:")
    print(f"  - 训练集: {len(train_dataset)} 个样本")
    print(f"  - 验证集: {len(val_dataset)} 个样本")
    
    # 统计训练集和验证集的标签分布
    print("\n=== 训练集标签统计 ===")
    train_label_counts = {}
    for idx in range(len(train_dataset)):
        _, label = train_dataset[idx]
        label = label.item()
        if label not in train_label_counts:
            train_label_counts[label] = 0
        train_label_counts[label] += 1
    
    for label_idx, count in train_label_counts.items():
        class_name = full_dataset.classes.get(label_idx, f"类别_{label_idx}")
        percentage = (count / len(train_dataset)) * 100
        print(f"  - {class_name} (标签 {label_idx}): {count} 个样本 ({percentage:.2f}%)")
    
    print("\n=== 验证集标签统计 ===")
    val_label_counts = {}
    for idx in range(len(val_dataset)):
        _, label = val_dataset[idx]
        label = label.item()
        if label not in val_label_counts:
            val_label_counts[label] = 0
        val_label_counts[label] += 1
    
    for label_idx, count in val_label_counts.items():
        class_name = full_dataset.classes.get(label_idx, f"类别_{label_idx}")
        percentage = (count / len(val_dataset)) * 100
        print(f"  - {class_name} (标签 {label_idx}): {count} 个样本 ({percentage:.2f}%)")
    print("=====================\n")
    
    # 3. 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # 4. 创建时序注意力模型
    print("初始化时序注意力模型...")
    
    # 获取特征维度
    n_features, time_steps = full_dataset.get_feature_dimensions()
    output_size = 4  # 4类道路分类
    
    print(f"时序特征维度: {n_features} × {time_steps}")
    print(f"输出类别数: {output_size}")
    
    model = create_temporal_attention_model(
        feature_dim=n_features,
        time_steps=time_steps,
        output_size=output_size,
        attention_dim=64,
        hidden_sizes=[128, 64, 32],
        dropout_rate=0.1
    )
    
    # 5. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
    # 6. 训练参数
    num_epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"使用设备: {device}")
    print(f"开始训练，共 {num_epochs} 个epoch...")
    
    # 7. 训练循环
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_accuracy = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_epoch_loss = 0.0
        train_correct_predictions = 0
        train_total_predictions = 0
        
        for batch_idx, (temporal_features, labels) in enumerate(train_loader):
            temporal_features = temporal_features.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(temporal_features)
            loss = criterion(outputs, labels)
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            train_total_predictions += labels.size(0)
            train_correct_predictions += (predicted == labels).sum().item()
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_epoch_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_epoch_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0
        
        # 按类别统计准确率
        class_correct = {}
        class_total = {}
        
        with torch.no_grad():
            for temporal_features, labels in val_loader:
                temporal_features = temporal_features.to(device)
                labels = labels.to(device)
                
                outputs = model(temporal_features)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                val_total_predictions += labels.size(0)
                val_correct_predictions += (predicted == labels).sum().item()
                
                # 统计每个类别的准确率
                for label, pred in zip(labels, predicted):
                    label_item = label.item()
                    pred_item = pred.item()
                    
                    # 初始化类别统计（如果尚未存在）
                    if label_item not in class_correct:
                        class_correct[label_item] = 0
                        class_total[label_item] = 0
                    
                    class_total[label_item] += 1
                    if pred_item == label_item:
                        class_correct[label_item] += 1
                
                val_epoch_loss += loss.item()
        
        # 更新学习率
        scheduler.step()
        
        # 计算平均损失和准确率
        train_avg_loss = train_epoch_loss / len(train_loader)
        train_accuracy = 100.0 * train_correct_predictions / train_total_predictions
        
        val_avg_loss = val_epoch_loss / len(val_loader)
        val_accuracy = 100.0 * val_correct_predictions / val_total_predictions
        
        train_losses.append(train_avg_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_avg_loss)
        val_accuracies.append(val_accuracy)
        
        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'  训练 - Loss: {train_avg_loss:.6f}, Accuracy: {train_accuracy:.2f}%')
            print(f'  验证 - Loss: {val_avg_loss:.6f}, Accuracy: {val_accuracy:.2f}%')
            
            # 打印每一类的验证准确率
            print(f'  验证集各类别准确率:')
            for label_idx in sorted(class_correct.keys()):
                if class_total[label_idx] > 0:
                    class_accuracy = 100.0 * class_correct[label_idx] / class_total[label_idx]
                    class_name = full_dataset.classes.get(label_idx, f"类别_{label_idx}")
                    print(f'    - {class_name}: {class_accuracy:.2f}% ({class_correct[label_idx]}/{class_total[label_idx]})')
    
    # 8. 保存最佳模型统计信息到weight目录
    model_save_path = "./weight/trained_temporal_attention_classifier.pth"
    
    # 确保weight目录存在
    os.makedirs("./weight", exist_ok=True)
    
    torch.save({
        'model_state_dict': best_model_state,
        'feature_dim': n_features,
        'time_steps': time_steps,
        'output_size': output_size,
        'class_mapping': full_dataset.classes,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_accuracy': best_val_accuracy,
    }, model_save_path)
    
    # 9. 计算每一类的准确率
    best_class_accuracy = {}
    for label_idx in sorted(class_correct.keys()):
        if class_total[label_idx] > 0:
            class_accuracy = 100.0 * class_correct[label_idx] / class_total[label_idx]
            class_name = full_dataset.classes.get(label_idx, f"类别_{label_idx}")
            best_class_accuracy[class_name] = round(class_accuracy, 2)
    
    # 10. 保存单一记录文件，包含所有要求的指标
    test_results = {
        # 训练数据和验证数据分布情况
        'train_label_distribution': train_label_counts,
        'val_label_distribution': val_label_counts,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        
        # 模型训练超参数
        'hyperparameters': {
            'feature_dim': n_features,
            'time_steps': time_steps,
            'output_size': output_size,
            'attention_dim': 64,
            'hidden_sizes': [128, 64, 32],
            'dropout_rate': 0.1,
            'learning_rate': 0.0001,
            'batch_size': 16,
            'num_epochs': num_epochs,
            'optimizer': 'Adam',
            'scheduler': 'StepLR',
            'weight_decay': 1e-5,
            'criterion': 'CrossEntropyLoss'
        },
        
        # 模型在验证集上每一类的预测准确率
        'class_accuracy': best_class_accuracy,
        
        # 最终总体准确率指标
        'final_val_accuracy': round(val_accuracies[-1], 2),
        'final_train_accuracy': round(train_accuracies[-1], 2),
        'best_val_accuracy': round(best_val_accuracy, 2),
        'final_val_loss': round(val_losses[-1], 6),
        'final_train_loss': round(train_losses[-1], 6),
        
        # 附加信息
        'class_mapping': full_dataset.classes,
        'timestamp': str(np.datetime64('now'))
    }
    
    with open('./weight/temporal_attention_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    print(f"训练完成！模型已保存到: {model_save_path}")
    print(f"测试结果已保存到: ./weight/temporal_attention_test_results.json")
    print(f"最佳验证准确率: {best_val_accuracy:.2f}%")
    print(f"最终训练准确率: {train_accuracies[-1]:.2f}%")
    print(f"最终验证准确率: {val_accuracies[-1]:.2f}%")
    
    return model, train_losses, train_accuracies, val_accuracies


def evaluate_temporal_model(model, dataset):
    """
    评估时序模型性能
    """
    model.eval()
    device = next(model.parameters()).device
    
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for temporal_features, labels in DataLoader(dataset, batch_size=16):
            temporal_features = temporal_features.to(device)
            labels = labels.to(device)
            
            outputs = model(temporal_features)
            _, predicted = torch.max(outputs.data, 1)
            
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    
    accuracy = 100.0 * correct_predictions / total_predictions
    print(f"评估准确率: {accuracy:.2f}%")
    
    return accuracy


if __name__ == "__main__":
    
    # 开始训练
    model, train_losses, train_accuracies, val_accuracies = train_temporal_attention_model()
    
    if model is not None:
        print("\n训练摘要:")
        print(f"- 最终训练损失: {train_losses[-1]:.6f}")
        print(f"- 最终训练准确率: {train_accuracies[-1]:.2f}%")
        print(f"- 最终验证准确率: {val_accuracies[-1]:.2f}%")
        print(f"- 模型已保存为: trained_temporal_attention_classifier.pth")
        
        # 可选：加载并评估模型
        print("\n加载模型进行评估...")
        try:
            checkpoint = torch.load("./weight/trained_temporal_attention_classifier.pth")
            model = create_temporal_attention_model(
                feature_dim=checkpoint['feature_dim'],
                time_steps=checkpoint['time_steps'],
                output_size=checkpoint['output_size']
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 使用合并时序数据集进行评估
            eval_dataset = CombinedTemporalTerrainDataset(
                DATASET_PATHS,
                time_window_seconds=1.0,
                sampling_rate=50
            )
            evaluate_temporal_model(model, eval_dataset)
        except Exception as e:
            print(f"评估模型时出错: {e}")
    else:
        print("训练失败")
