import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    """
    时序注意力机制
    用于处理n×T的时序特征矩阵
    """
    def __init__(self, feature_dim, time_steps, attention_dim=64):
        """
        初始化时序注意力层
        
        参数:
            feature_dim (int): 特征维度 (n)
            time_steps (int): 时序长度 (T)
            attention_dim (int): 注意力隐藏维度
        """
        super(TemporalAttention, self).__init__()
        
        self.feature_dim = feature_dim
        self.time_steps = time_steps
        self.attention_dim = attention_dim
        
        # 注意力机制 - 修正：使用相同的输入输出维度
        self.query = nn.Linear(feature_dim, attention_dim)
        self.key = nn.Linear(feature_dim, attention_dim)
        self.value = nn.Linear(feature_dim, feature_dim)  # 保持原始特征维度
        
        # 注意力缩放因子
        self.scale = torch.sqrt(torch.tensor(attention_dim, dtype=torch.float32))
        
        # 输出投影 - 修正：从feature_dim到feature_dim
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入时序特征，形状为 (batch_size, feature_dim, time_steps)
            
        返回:
            torch.Tensor: 注意力加权的特征，形状为 (batch_size, feature_dim)
        """
        batch_size = x.size(0)
        
        # 转置为 (batch_size, time_steps, feature_dim)
        x_transposed = x.transpose(1, 2)
        
        # 计算Q, K, V
        Q = self.query(x_transposed)  # (batch_size, time_steps, attention_dim)
        K = self.key(x_transposed)    # (batch_size, time_steps, attention_dim)
        V = self.value(x_transposed)  # (batch_size, time_steps, feature_dim)
        
        # 计算注意力分数
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (batch_size, time_steps, time_steps)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, time_steps, time_steps)
        
        # 应用注意力权重
        attended_features = torch.bmm(attention_weights, V)  # (batch_size, time_steps, feature_dim)
        
        # 全局平均池化得到最终特征
        global_features = attended_features.mean(dim=1)  # (batch_size, feature_dim)
        
        # 输出投影
        output = self.output_proj(global_features)  # (batch_size, feature_dim)
        
        return output


class TemporalAttentionMLP(nn.Module):
    """
    时序注意力 + MLP分类模型
    先使用时序注意力提取时序特征，然后使用MLP进行分类
    """
    def __init__(self, feature_dim, time_steps, hidden_sizes=[128, 64, 32], output_size=4, 
                 attention_dim=64, dropout_rate=0.1):
        """
        初始化时序注意力MLP模型
        
        参数:
            feature_dim (int): 输入特征维度 (n)
            time_steps (int): 时序长度 (T)
            hidden_sizes (list): MLP隐藏层大小列表
            output_size (int): 输出类别数
            attention_dim (int): 注意力隐藏维度
            dropout_rate (float): dropout率
        """
        super(TemporalAttentionMLP, self).__init__()
        
        self.feature_dim = feature_dim
        self.time_steps = time_steps
        
        # 时序注意力层
        self.temporal_attention = TemporalAttention(
            feature_dim=feature_dim,
            time_steps=time_steps,
            attention_dim=attention_dim
        )
        
        # MLP分类器
        mlp_layers = []
        prev_size = feature_dim
        
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
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入时序特征，形状为 (batch_size, feature_dim, time_steps)
            
        返回:
            torch.Tensor: 分类输出，形状为 (batch_size, output_size)
        """
        # 时序注意力提取特征
        temporal_features = self.temporal_attention(x)  # (batch_size, feature_dim)
        
        # MLP分类
        output = self.mlp_classifier(temporal_features)  # (batch_size, output_size)
        
        return output


def create_temporal_attention_model(feature_dim, time_steps, hidden_sizes=[128, 64, 32], 
                                   output_size=4, attention_dim=64, dropout_rate=0.1):
    """
    创建时序注意力MLP模型的便捷函数
    
    参数:
        feature_dim (int): 输入特征维度 (n)
        time_steps (int): 时序长度 (T)
        hidden_sizes (list): MLP隐藏层大小列表
        output_size (int): 输出类别数
        attention_dim (int): 注意力隐藏维度
        dropout_rate (float): dropout率
    
    返回:
        TemporalAttentionMLP: 初始化的时序注意力MLP模型
    """
    model = TemporalAttentionMLP(
        feature_dim=feature_dim,
        time_steps=time_steps,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        attention_dim=attention_dim,
        dropout_rate=dropout_rate
    )
    return model


if __name__ == "__main__":
    # 测试模型
    feature_dim = 18  # 18个特征
    time_steps = 50   # 50个时间步
    
    model = create_temporal_attention_model(
        feature_dim=feature_dim,
        time_steps=time_steps,
        output_size=4
    )
    
    print("时序注意力MLP模型结构:")
    print(model)
    
    # 测试前向传播
    batch_size = 5
    test_input = torch.randn(batch_size, feature_dim, time_steps)
    output = model(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    
    # 打印参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数数量: {total_params:,}")
