import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    """
    多层感知机模型
    """
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.1):
        """
        初始化MLP模型
        
        参数:
            input_size (int): 输入特征维度
            hidden_sizes (list): 隐藏层大小列表
            output_size (int): 输出维度
            dropout_rate (float): dropout率
        """
        super(MLP, self).__init__()
        
        # 创建网络层
        layers = []
        prev_size = input_size
        
        # 添加隐藏层
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # 添加输出层
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
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
        """前向传播"""
        return self.network(x)


def create_mlp_model(input_size, hidden_sizes=[128, 64, 32], output_size=1, dropout_rate=0.1):
    """
    创建MLP模型的便捷函数
    
    参数:
        input_size (int): 输入特征维度
        hidden_sizes (list): 隐藏层大小列表
        output_size (int): 输出维度
        dropout_rate (float): dropout率
    
    返回:
        MLP: 初始化的MLP模型
    """
    model = MLP(input_size, hidden_sizes, output_size, dropout_rate)
    return model


if __name__ == "__main__":
    # 测试模型
    model = create_mlp_model(input_size=10, output_size=1)
    print("模型结构:")
    print(model)
    
    # 测试前向传播
    test_input = torch.randn(5, 10)
    output = model(test_input)
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
