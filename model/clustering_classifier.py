import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class ClusteringFeatureExtractor:
    """
    聚类特征提取器
    使用K-means聚类提取输入特征的高级表示
    """
    def __init__(self, n_clusters=8, random_state=42):
        """
        初始化聚类特征提取器
        
        参数:
            n_clusters (int): 聚类中心数量
            random_state (int): 随机种子
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, features):
        """
        训练聚类模型
        
        参数:
            features (numpy.ndarray): 训练特征，形状为 (n_samples, n_features)
        """
        # 标准化特征
        features_scaled = self.scaler.fit_transform(features)
        
        # 训练K-means聚类
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.kmeans.fit(features_scaled)
        self.is_fitted = True
    
    def transform(self, features):
        """
        转换特征为聚类特征
        
        参数:
            features (numpy.ndarray): 输入特征，形状为 (n_samples, n_features)
            
        返回:
            numpy.ndarray: 聚类特征，形状为 (n_samples, n_clusters + 1)
        """
        if not self.is_fitted:
            raise ValueError("聚类模型尚未训练，请先调用fit方法")
        
        # 标准化特征
        features_scaled = self.scaler.transform(features)
        
        # 计算到每个聚类中心的距离
        distances = self.kmeans.transform(features_scaled)  # (n_samples, n_clusters)
        
        # 获取聚类标签
        cluster_labels = self.kmeans.predict(features_scaled)  # (n_samples,)
        
        # 创建聚类特征：距离 + 聚类标签的one-hot编码
        cluster_one_hot = np.eye(self.n_clusters)[cluster_labels]  # (n_samples, n_clusters)
        
        # 合并特征：距离 + one-hot编码
        cluster_features = np.concatenate([distances, cluster_one_hot], axis=1)  # (n_samples, n_clusters * 2)
        
        return cluster_features
    
    def fit_transform(self, features):
        """训练并转换特征"""
        self.fit(features)
        return self.transform(features)


class ClusteringMLP(nn.Module):
    """
    聚类 + MLP分类模型
    先使用聚类提取高级特征，然后使用MLP进行分类
    """
    def __init__(self, input_size, n_clusters=8, hidden_sizes=[128, 64, 32], 
                 output_size=4, dropout_rate=0.1):
        """
        初始化聚类MLP模型
        
        参数:
            input_size (int): 输入特征维度
            n_clusters (int): 聚类中心数量
            hidden_sizes (list): MLP隐藏层大小列表
            output_size (int): 输出类别数
            dropout_rate (float): dropout率
        """
        super(ClusteringMLP, self).__init__()
        
        self.input_size = input_size
        self.n_clusters = n_clusters
        self.cluster_feature_dim = n_clusters * 2  # 距离 + one-hot编码
        
        # 聚类特征提取器（在训练过程中动态训练）
        self.clustering_extractor = None
        
        # MLP分类器
        mlp_layers = []
        prev_size = self.cluster_feature_dim
        
        # 添加隐藏层
        for hidden_size in hidden_sizes:
            mlp_layers.append(nn.Linear(prev_size, hidden_size))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # 添加输出层
        mlp_layers.append(nn.Linear(prev_size, output_size))
        
        self.mlp_classifier = nn.Sequential(*mlp_layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def fit_clustering(self, features_numpy):
        """
        训练聚类特征提取器
        
        参数:
            features_numpy (numpy.ndarray): 训练特征，形状为 (n_samples, input_size)
        """
        self.clustering_extractor = ClusteringFeatureExtractor(n_clusters=self.n_clusters)
        self.clustering_extractor.fit(features_numpy)
    
    def extract_cluster_features(self, features_numpy):
        """
        提取聚类特征
        
        参数:
            features_numpy (numpy.ndarray): 输入特征，形状为 (n_samples, input_size)
            
        返回:
            numpy.ndarray: 聚类特征，形状为 (n_samples, cluster_feature_dim)
        """
        if self.clustering_extractor is None:
            raise ValueError("聚类特征提取器尚未训练，请先调用fit_clustering方法")
        
        return self.clustering_extractor.transform(features_numpy)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入特征，形状为 (batch_size, input_size)
            
        返回:
            torch.Tensor: 分类输出，形状为 (batch_size, output_size)
        """
        # 注意：聚类特征提取需要在训练循环中单独处理
        # 这里假设输入已经是聚类特征
        output = self.mlp_classifier(x)  # (batch_size, output_size)
        
        return output


class EndToEndClusteringClassifier:
    """
    端到端的聚类分类器
    封装了聚类特征提取和MLP分类的完整流程
    """
    def __init__(self, input_size, n_clusters=8, hidden_sizes=[128, 64, 32], 
                 output_size=4, dropout_rate=0.1):
        """
        初始化端到端聚类分类器
        
        参数:
            input_size (int): 输入特征维度
            n_clusters (int): 聚类中心数量
            hidden_sizes (list): MLP隐藏层大小列表
            output_size (int): 输出类别数
            dropout_rate (float): dropout率
        """
        self.input_size = input_size
        self.n_clusters = n_clusters
        self.cluster_feature_dim = n_clusters * 2
        
        # 聚类MLP模型
        self.model = ClusteringMLP(
            input_size=input_size,
            n_clusters=n_clusters,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            dropout_rate=dropout_rate
        )
        
        # 训练状态
        self.is_clustering_fitted = False
    
    def fit_clustering(self, features_numpy):
        """
        训练聚类特征提取器
        
        参数:
            features_numpy (numpy.ndarray): 训练特征，形状为 (n_samples, input_size)
        """
        self.model.fit_clustering(features_numpy)
        self.is_clustering_fitted = True
    
    def prepare_features(self, features_numpy):
        """
        准备聚类特征
        
        参数:
            features_numpy (numpy.ndarray): 原始特征，形状为 (n_samples, input_size)
            
        返回:
            torch.Tensor: 聚类特征张量
        """
        if not self.is_clustering_fitted:
            raise ValueError("聚类特征提取器尚未训练")
        
        cluster_features = self.model.extract_cluster_features(features_numpy)
        return torch.tensor(cluster_features, dtype=torch.float32)
    
    def forward(self, x_cluster_features):
        """
        前向传播（使用聚类特征）
        
        参数:
            x_cluster_features (torch.Tensor): 聚类特征，形状为 (batch_size, cluster_feature_dim)
            
        返回:
            torch.Tensor: 分类输出
        """
        return self.model(x_cluster_features)


def create_clustering_classifier(input_size, n_clusters=8, hidden_sizes=[128, 64, 32], 
                                output_size=4, dropout_rate=0.1):
    """
    创建聚类分类器的便捷函数
    
    参数:
        input_size (int): 输入特征维度
        n_clusters (int): 聚类中心数量
        hidden_sizes (list): MLP隐藏层大小列表
        output_size (int): 输出类别数
        dropout_rate (float): dropout率
    
    返回:
        EndToEndClusteringClassifier: 初始化的聚类分类器
    """
    classifier = EndToEndClusteringClassifier(
        input_size=input_size,
        n_clusters=n_clusters,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        dropout_rate=dropout_rate
    )
    return classifier


if __name__ == "__main__":
    # 测试聚类分类器
    input_size = 18  # 18个特征
    
    classifier = create_clustering_classifier(
        input_size=input_size,
        output_size=4
    )
    
    print("聚类分类器结构:")
    print(classifier.model)
    
    # 模拟训练聚类特征提取器
    print("\n模拟聚类特征提取器训练...")
    n_samples = 100
    test_features = np.random.randn(n_samples, input_size)
    classifier.fit_clustering(test_features)
    
    # 测试特征转换
    test_batch = np.random.randn(5, input_size)
    cluster_features = classifier.prepare_features(test_batch)
    
    print(f"原始特征形状: {test_batch.shape}")
    print(f"聚类特征形状: {cluster_features.shape}")
    
    # 测试前向传播
    output = classifier.forward(cluster_features)
    print(f"分类输出形状: {output.shape}")
    
    # 打印参数数量
    total_params = sum(p.numel() for p in classifier.model.parameters())
    print(f"MLP部分参数数量: {total_params:,}")
