import sys
import os
import json
import numpy as np
import random
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from data.dataset import TerrainDataset
from model.svm import SVMClassifier


class CombinedTerrainDataset:
    """
    合并多个TerrainDataset的数据集类，用于SVM训练
    """
    def __init__(self, dataset_paths):
        self.datasets = []
        self.features = []
        self.labels = []
        
        for dataset_path in dataset_paths:
            try:
                dataset = TerrainDataset(dataset_path)
                self.datasets.append(dataset)
                
                # 收集所有样本的特征和标签
                for sample in dataset.samples:
                    if sample['features'] is not None:
                        self.features.append(sample['features'])
                        self.labels.append(sample['label'])
                
                print(f"成功加载数据集: {dataset_path}, 样本数: {len(dataset)}")
            except Exception as e:
                print(f"加载数据集 {dataset_path} 失败: {e}")
        
        if len(self.features) == 0:
            raise ValueError("没有成功加载任何数据集")
        
        # 转换为numpy数组
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        
        # 统计标签分布
        self.label_counts = {}
        for label in self.labels:
            if label not in self.label_counts:
                self.label_counts[label] = 0
            self.label_counts[label] += 1
        
        # 打乱数据
        self._shuffle_data()
        
        print(f"总共加载 {len(self.datasets)} 个数据集，总样本数: {len(self.features)}")
        
        # 打印标签统计信息
        self.print_label_statistics()
    
    def _shuffle_data(self):
        """打乱特征和标签的顺序"""
        indices = np.random.permutation(len(self.features))
        self.features = self.features[indices]
        self.labels = self.labels[indices]
    
    
    def print_label_statistics(self):
        """打印数据集的标签统计信息"""
        print("\n=== 数据集标签统计 ===")
        if not self.datasets:
            print("没有可用的类别映射信息")
            return
        
        class_mapping = self.datasets[0].classes
        total_samples = len(self.features)
        
        print(f"总样本数: {total_samples}")
        print("各类别样本分布:")
        
        for label_idx, count in self.label_counts.items():
            class_name = class_mapping.get(label_idx, f"未知类别_{label_idx}")
            percentage = (count / total_samples) * 100
            print(f"  - {class_name} (标签 {label_idx}): {count} 个样本 ({percentage:.2f}%)")
        
        print("=====================\n")
    
    def split_data(self, train_ratio=0.8):
        """分割数据集为训练集和测试集"""
        split_idx = int(len(self.features) * train_ratio)
        
        X_train = self.features[:split_idx]
        y_train = self.labels[:split_idx]
        X_test = self.features[split_idx:]
        y_test = self.labels[split_idx:]
        
        print(f"数据集分割:")
        print(f"  - 训练集: {len(X_train)} 个样本")
        print(f"  - 测试集: {len(X_test)} 个样本")
        
        return X_train, X_test, y_train, y_test
    
    @property
    def classes(self):
        """返回类别映射（使用第一个数据集的类别映射）"""
        if self.datasets:
            return self.datasets[0].classes
        return {}


# 定义所有数据集路径（与MLP训练相同）
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


def train_svm_classifier():
    """
    训练SVM分类器进行地形分类
    """
    print("开始训练SVM分类器...")
    
    # 1. 创建合并数据集
    print("创建合并数据集...")
    try:
        full_dataset = CombinedTerrainDataset(DATASET_PATHS)
    except Exception as e:
        print(f"创建数据集失败: {e}")
        return None, None
    
    if len(full_dataset.features) == 0:
        print("没有有效的训练样本")
        return None, None
    
    # 2. 分割数据集为训练集和测试集 (80% 训练, 20% 测试)
    X_train, X_test, y_train, y_test = full_dataset.split_data(train_ratio=0.8)
    
    # 4. 创建并训练SVM模型
    print("初始化SVM模型...")
    
    # 使用新的SVMClassifier类
    svm_model = SVMClassifier(
        kernel='rbf',           # 径向基函数核
        C=1.0,                  # 正则化参数
        gamma='scale',          # 核系数
        probability=True,       # 启用概率预测
        random_state=42         # 随机种子
    )
    
    print("开始训练SVM模型...")
    svm_model.fit(X_train, y_train, full_dataset.classes)
    
    # 5. 在训练集和测试集上评估模型
    print("\n=== 模型评估 ===")
    
    # 训练集评估
    y_train_pred = svm_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"训练集准确率: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    
    # 测试集评估
    y_test_pred = svm_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"测试集准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # 计算每一类的准确率
    class_accuracy = {}
    class_names = [full_dataset.classes.get(i, f"类别_{i}") for i in range(4)]
    
    for label_idx in range(4):
        mask = y_test == label_idx
        if np.sum(mask) > 0:
            class_acc = accuracy_score(y_test[mask], y_test_pred[mask])
            class_name = full_dataset.classes.get(label_idx, f"类别_{label_idx}")
            class_accuracy[class_name] = round(class_acc * 100, 2)
            print(f"  - {class_name} 准确率: {class_acc:.4f} ({class_acc*100:.2f}%)")
    
    # 6. 打印详细分类报告
    print("\n=== 详细分类报告 ===")
    print(classification_report(y_test, y_test_pred, target_names=class_names))
    
    # 7. 打印混淆矩阵
    print("\n=== 混淆矩阵 ===")
    cm = confusion_matrix(y_test, y_test_pred)
    print("预测\\真实", end="")
    for name in class_names:
        print(f"\t{name}", end="")
    print()
    
    for i, true_name in enumerate(class_names):
        print(f"{true_name}", end="")
        for j in range(len(class_names)):
            print(f"\t{cm[i, j]}", end="")
        print()
    
    # 8. 保存模型和相关信息到weight目录
    model_save_path = "./weight/trained_svm_classifier.pkl"
    info_save_path = "./weight/svm_model_info.json"
    
    # 确保weight目录存在
    os.makedirs("./weight", exist_ok=True)
    
    # 使用SVMClassifier的保存方法
    svm_model.save_model(model_save_path, info_save_path)
    
    # 统计训练数据和验证数据分布
    train_label_counts = {}
    for label in y_train:
        if label not in train_label_counts:
            train_label_counts[label] = 0
        train_label_counts[label] += 1
    
    val_label_counts = {}
    for label in y_test:
        if label not in val_label_counts:
            val_label_counts[label] = 0
        val_label_counts[label] += 1
    
    # 转换为类别名称的分布
    train_label_distribution = {}
    val_label_distribution = {}
    
    for label_idx, count in train_label_counts.items():
        class_name = full_dataset.classes.get(label_idx, f"类别_{label_idx}")
        train_label_distribution[class_name] = count
    
    for label_idx, count in val_label_counts.items():
        class_name = full_dataset.classes.get(label_idx, f"类别_{label_idx}")
        val_label_distribution[class_name] = count
    
    # 9. 保存单一记录文件，包含所有必需信息
    test_results = {
        # 训练数据和验证数据分布情况
        'train_label_distribution': train_label_distribution,
        'val_label_distribution': val_label_distribution,
        'train_samples': len(X_train),
        'val_samples': len(X_test),
        
        # 模型训练超参数
        'hyperparameters': {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'probability': True,
            'random_state': 42
        },
        
        # 模型在验证集上每一类的预测准确率
        'class_accuracy': class_accuracy,
        
        # 最终总体准确率指标
        'final_val_accuracy': test_accuracy,
        'final_train_accuracy': train_accuracy,
        
        # 其他信息
        'class_mapping': full_dataset.classes,
        'feature_dim': X_train.shape[1],
        'num_classes': len(full_dataset.classes),
        'timestamp': str(np.datetime64('now'))
    }
    
    with open('./weight/svm_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n模型训练完成！")
    print(f"SVM模型已保存到: {model_save_path}")
    print(f"单一记录文件已保存到: ./weight/svm_test_results.json")
    print(f"最终测试准确率: {test_accuracy*100:.2f}%")
    
    return svm_model, test_accuracy


def evaluate_svm_model(model, dataset_paths):
    """
    评估SVM模型性能
    """
    print("\n=== 模型评估 ===")
    
    # 创建评估数据集
    try:
        eval_dataset = CombinedTerrainDataset(dataset_paths)
        
        # 预测
        y_pred = model.predict(eval_dataset.features)
        accuracy = accuracy_score(eval_dataset.labels, y_pred)
        
        print(f"评估准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # 打印分类报告
        class_names = [eval_dataset.classes.get(i, f"类别_{i}") for i in range(4)]
        print("\n详细分类报告:")
        print(classification_report(eval_dataset.labels, y_pred, target_names=class_names))
        
        return accuracy
        
    except Exception as e:
        print(f"评估模型时出错: {e}")
        return None


def predict_terrain(model, features, class_mapping):
    """
    使用训练好的SVM模型预测地形类型
    
    参数:
        model: 训练好的SVM模型
        features: 特征向量
        class_mapping: 类别映射字典
    
    返回:
        dict: 包含预测结果和概率的字典
    """
    # 预测
    prediction = model.predict([features])[0]
    probabilities = model.predict_proba([features])[0]
    
    # 获取类别名称
    class_name = class_mapping.get(prediction, f"未知类别_{prediction}")
    
    # 创建概率字典
    prob_dict = {}
    for i, prob in enumerate(probabilities):
        prob_name = class_mapping.get(i, f"类别_{i}")
        prob_dict[prob_name] = prob
    
    return {
        'predicted_class': class_name,
        'predicted_label': prediction,
        'probabilities': prob_dict,
        'confidence': probabilities[prediction]
    }


if __name__ == "__main__":
    
    # 开始训练
    svm_model, test_accuracy = train_svm_classifier()
    
    if svm_model is not None:
        print("\n训练摘要:")
        print(f"- 最终测试准确率: {test_accuracy*100:.2f}%")
        print(f"- 模型已保存为: trained_svm_classifier.pkl")
        
        # 可选：加载并评估模型
        print("\n加载模型进行评估...")
        try:
            # 使用SVMClassifier的加载方法
            loaded_model = SVMClassifier()
            loaded_model.load_model("./weight/trained_svm_classifier.pkl", "./weight/svm_model_info.json")
            
            # 加载测试结果信息（使用新的单一记录文件）
            with open('./weight/svm_test_results.json', 'r') as f:
                test_results = json.load(f)
            
            # 使用相同的数据集进行评估
            evaluate_svm_model(loaded_model, DATASET_PATHS)
            
            # 示例：使用模型进行单样本预测
            print("\n=== 单样本预测示例 ===")
            # 获取一个样本进行预测演示
            sample_dataset = CombinedTerrainDataset([DATASET_PATHS[0]])
            if len(sample_dataset.features) > 0:
                sample_features = sample_dataset.features[0]
                sample_label = sample_dataset.labels[0]
                
                result = predict_terrain(
                    loaded_model, 
                    sample_features, 
                    test_results['class_mapping']
                )
                
                true_class = test_results['class_mapping'].get(sample_label, f"类别_{sample_label}")
                print(f"真实类别: {true_class}")
                print(f"预测类别: {result['predicted_class']}")
                print(f"置信度: {result['confidence']:.4f}")
                print("各类别概率:")
                for class_name, prob in result['probabilities'].items():
                    print(f"  - {class_name}: {prob:.4f}")
                    
        except Exception as e:
            print(f"评估模型时出错: {e}")
    else:
        print("训练失败")
