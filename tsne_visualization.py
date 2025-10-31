import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from data.temporal_dataset import TemporalTerrainDataset
from model.temporal_attention import create_temporal_attention_model
import os
import json
import sys


class TemporalAttentionFeatureExtractor(nn.Module):
    """
    Temporal attention feature extractor for extracting features after temporal attention encoding
    """
    def __init__(self, original_model):
        super(TemporalAttentionFeatureExtractor, self).__init__()
        self.original_model = original_model
        
    def forward(self, x):
        """
        Extract features after temporal attention encoding
        
        Args:
            x (torch.Tensor): Input temporal features with shape (batch_size, feature_dim, time_steps)
            
        Returns:
            torch.Tensor: Temporal attention encoded features with shape (batch_size, feature_dim)
        """
        # Use temporal attention layer to extract features
        temporal_features = self.original_model.temporal_attention(x)
        return temporal_features


def extract_temporal_attention_features(model, dataloader, device):
    """
    提取数据集的时序注意力编码特征
    
    参数:
        model: 训练好的时序注意力模型
        dataloader: 数据加载器
        device: 设备
    
    返回:
        features: 时序注意力编码特征 (n_samples, feature_dim)
        labels: 对应的标签 (n_samples,)
    """
    feature_extractor = TemporalAttentionFeatureExtractor(model)
    feature_extractor.to(device)
    feature_extractor.eval()
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(device)
            
            # 提取时序注意力编码特征
            batch_attention_features = feature_extractor(batch_features)
            
            all_features.append(batch_attention_features.cpu().numpy())
            all_labels.append(batch_labels.numpy())
    
    # 合并所有批次的特征和标签
    features = np.vstack(all_features)
    labels = np.hstack(all_labels)
    
    print(f"提取的时序注意力特征形状: {features.shape}")
    print(f"标签形状: {labels.shape}")
    
    return features, labels


def perform_tsne_visualization(features, labels, class_mapping, title="t-SNE Visualization"):
    """
    使用t-SNE进行降维可视化
    
    参数:
        features: 特征矩阵 (n_samples, n_features)
        labels: 标签数组 (n_samples,)
        class_mapping: 类别映射字典
        title: 图表标题
    """
    print("Starting t-SNE dimensionality reduction...")
    
    # 如果特征维度太高，先使用PCA降维到50维
    if features.shape[1] > 50:
        print(f"High feature dimensionality ({features.shape[1]}), using PCA to reduce to 50 dimensions first...")
        pca = PCA(n_components=50, random_state=42)
        features = pca.fit_transform(features)
        print(f"Feature shape after PCA: {features.shape}")
    
    # 执行t-SNE降维到2维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(features)
    
    print(f"Feature shape after t-SNE: {features_2d.shape}")
    
    # 创建可视化
    plt.figure(figsize=(12, 10))
    
    # 获取唯一的标签和对应的类别名称
    unique_labels = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    # 为每个类别绘制散点
    for i, label in enumerate(unique_labels):
        mask = labels == label
        class_name = class_mapping.get(label, f"Class_{label}")
        
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[colors[i]], label=class_name, alpha=0.7, s=30)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    plt.figtext(0.02, 0.02, 
               f"Total samples: {len(features)}\nFeature dimensions: {features.shape[1]}\nNumber of classes: {len(unique_labels)}", 
               fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('temporal_attention_tsne_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return features_2d


def analyze_feature_distribution(features, labels, class_mapping):
    """
    分析特征分布统计信息
    """
    print("\n=== Temporal Attention Feature Distribution Analysis ===")
    print(f"Feature matrix shape: {features.shape}")
    print(f"Feature statistics:")
    print(f"  - Mean: {np.mean(features):.4f}")
    print(f"  - Standard deviation: {np.std(features):.4f}")
    print(f"  - Minimum: {np.min(features):.4f}")
    print(f"  - Maximum: {np.max(features):.4f}")
    
    # 按类别分析特征统计
    unique_labels = np.unique(labels)
    print(f"\nFeature statistics by class:")
    for label in unique_labels:
        mask = labels == label
        class_features = features[mask]
        class_name = class_mapping.get(label, f"Class_{label}")
        
        print(f"  {class_name} (Label {label}):")
        print(f"    - Number of samples: {len(class_features)}")
        print(f"    - Feature mean: {np.mean(class_features):.4f}")
        print(f"    - Feature standard deviation: {np.std(class_features):.4f}")


def load_all_temporal_datasets():
    """
    加载所有时序数据集
    
    返回:
        combined_dataset: 合并的数据集
    """
    # 数据集路径列表（简化版本，只包含几个数据集用于测试）
    DATASET_PATHS = [
        "/home/xie/xzk/Graduation/Terrain/dataset/dataset/2024-09-22-16-12-28",
        "/home/xie/xzk/Graduation/Terrain/dataset/dataset/2024-09-22-16-40-20",
        "/home/xie/xzk/Graduation/Terrain/dataset/dataset/2024-09-26-15-35-26",
        "/home/xie/xzk/Graduation/Terrain/dataset/dataset/2025-02-25-11-43-52",
        "/home/xie/xzk/Graduation/Terrain/dataset/dataset/2025-02-25-11-45-15",
        "/home/xie/xzk/Graduation/Terrain/dataset/dataset/2025-02-25-11-45-55",
        "/home/xie/xzk/Graduation/Terrain/dataset/dataset/2025-02-25-11-47-21",
        "/home/xie/xzk/Graduation/Terrain/dataset/dataset/2025-02-25-11-47-45",
        "/home/xie/xzk/Graduation/Terrain/dataset/dataset/2025-02-25-14-26-20",
        "/home/xie/xzk/Graduation/Terrain/dataset/dataset/2025-02-25-14-27-13",
        "/home/xie/xzk/Graduation/Terrain/dataset/dataset/2025-02-25-10-54-50",
        "/home/xie/xzk/Graduation/Terrain/dataset/dataset/2025-02-25-10-55-52",
        "/home/xie/xzk/Graduation/Terrain/dataset/dataset/2025-02-25-10-57-01",
        "/home/xie/xzk/Graduation/Terrain/dataset/dataset/2025-02-25-10-58-08",
        "/home/xie/xzk/Graduation/Terrain/dataset/dataset/2025-02-25-11-03-50",
        "/home/xie/xzk/Graduation/Terrain/dataset/dataset/2025-02-25-11-04-48",
        "/home/xie/xzk/Graduation/Terrain/dataset/dataset/2025-02-25-11-08-24",
        "/home/xie/xzk/Graduation/Terrain/dataset/dataset/2025-02-25-11-08-54",
        "/home/xie/xzk/Graduation/Terrain/dataset/dataset/2025-02-25-11-21-54"
    ]
    
    all_samples = []
    
    for dataset_path in DATASET_PATHS:
        try:
            print(f"Loading dataset: {dataset_path}")
            dataset = TemporalTerrainDataset(dataset_path, time_window_seconds=1.0, sampling_rate=50)
            
            # 获取数据集中的所有样本
            for i in range(len(dataset)):
                features, label = dataset[i]
                all_samples.append((features, label))
                
            print(f"  Successfully loaded {len(dataset)} samples")
            
        except Exception as e:
            print(f"  Failed to load dataset: {e}")
            continue
    
    print(f"Total loaded {len(all_samples)} samples")
    
    # 创建自定义数据集类
    class CombinedTemporalDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.samples = samples
            
        def __len__(self):
            return len(self.samples)
            
        def __getitem__(self, idx):
            return self.samples[idx]
    
    return CombinedTemporalDataset(all_samples)


def main():
    """
    主函数：加载时序注意力模型，提取编码特征，进行t-SNE可视化
    """
    print("Starting t-SNE visualization analysis of temporal attention features...")
    
    # 1. 检查模型文件是否存在
    model_path = "weight/attention/trained_temporal_attention_classifier.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} does not exist")
        print("Please run train_temporal_attention.py to train the model first")
        return
    
    # 2. 加载模型
    print("Loading trained temporal attention model...")
    try:
        checkpoint = torch.load(model_path)
        feature_dim = checkpoint['feature_dim']
        time_steps = checkpoint['time_steps']
        output_size = checkpoint['output_size']
        class_mapping = checkpoint['class_mapping']
        
        model = create_temporal_attention_model(
            feature_dim=feature_dim,
            time_steps=time_steps,
            output_size=output_size
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded successfully:")
        print(f"  - Feature dimension: {feature_dim}")
        print(f"  - Time steps: {time_steps}")
        print(f"  - Output size: {output_size}")
        print(f"  - Class mapping: {class_mapping}")
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # 3. 加载所有数据集
    print("Loading all temporal datasets...")
    try:
        combined_dataset = load_all_temporal_datasets()
        dataloader = torch.utils.data.DataLoader(combined_dataset, batch_size=32, shuffle=False)
        
        print(f"Dataset loaded successfully:")
        print(f"  - Total samples: {len(combined_dataset)}")
        print(f"  - Number of classes: {len(class_mapping)}")
        
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
    
    # 4. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")
    
    # 5. 提取时序注意力编码特征
    print("\nExtracting temporal attention encoded features...")
    features, labels = extract_temporal_attention_features(model, dataloader, device)
    
    # 6. 分析特征分布
    analyze_feature_distribution(features, labels, class_mapping)
    
    # 7. 执行t-SNE可视化
    print("\nExecuting t-SNE visualization...")
    features_2d = perform_tsne_visualization(
        features, labels, class_mapping, 
        title="t-SNE Visualization of Temporal Attention Encoded Features"
    )
    
    # 8. 保存结果
    results = {
        'features_2d': features_2d.tolist(),
        'labels': labels.tolist(),
        'class_mapping': class_mapping,
        'original_features_shape': features.shape,
        'model_info': {
            'feature_dim': feature_dim,
            'time_steps': time_steps,
            'output_size': output_size
        }
    }
    
    with open('temporal_attention_tsne_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nVisualization completed!")
    print(f"- t-SNE results saved to: temporal_attention_tsne_results.json")
    print(f"- Visualization chart saved to: temporal_attention_tsne_visualization.png")
    print(f"- Feature dimensions: {features.shape[1]} -> 2")
    print(f"- Total samples: {len(features)}")
    print(f"- Class distribution:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        class_name = class_mapping.get(label, f"Class_{label}")
        print(f"  {class_name}: {count} samples")


if __name__ == "__main__":
    main()
