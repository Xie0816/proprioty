import numpy as np
import torch
import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_loader import Proprio_Loader
from torch.utils.data import Dataset
from collections import deque


class TemporalTerrainDataset(Dataset):
    """
    时序地形分类数据集类
    处理单个数据集文件夹，提取时序传感器信息作为输入
    考虑1秒内的传感器输入，聚合为n×T的矩阵
    输出为四类道路分类：Dirt_Road、Asphalt_Road、Cement_Road、Gravel_Road
    """
    
    def __init__(self, dataset_path, time_window_seconds=1.0, sampling_rate=50):
        """初始化时序数据集"""
        self.dataset_path = dataset_path
        self.loader = Proprio_Loader(dataset_path)
        self.roi_data = None
        self.proprio_data = None
        self.samples = []
        self.classes = {
            'Dirt_Road': 0,
            'Asphalt_Road': 1, 
            'Cement_Road': 2,
            'Gravel_Road': 3
        }
        
        # 时序参数
        self.time_window_seconds = time_window_seconds
        self.sampling_rate = sampling_rate
        self.time_window_frames = int(time_window_seconds * sampling_rate)
        
        # 历史时序数据队列
        self.temporal_buffer = deque(maxlen=self.time_window_frames)
        
        # 加载数据并准备样本
        self._load_data()
        self._prepare_samples()
    
    def _load_data(self):
        """加载数据"""
        print(f"加载时序数据集: {self.dataset_path}")
        
        self.roi_data = self.loader.load_roi_frame_json()
        self.proprio_data = self.loader.load_proprio_seg_infos_json()
        
        if not self.roi_data or not self.proprio_data:
            raise ValueError("数据加载失败")
        
        print(f"成功加载数据:")
        print(f"  - roi_frame: {len(self.roi_data)} 个类别")
        print(f"  - proprio_seg_infos: {len(self.proprio_data)} 个帧")
        print(f"  - 时序窗口: {self.time_window_seconds}秒 ({self.time_window_frames}帧)")
    
    def _extract_temporal_features(self, frame_id):
        """
        从时序数据中提取特征矩阵
        返回n×T的矩阵，其中n是特征数量，T是时序长度
        
        参数:
            frame_id: 当前帧ID
            
        返回:
            numpy.array: n×T的特征矩阵
        """
        try:
            # 获取当前帧的时序特征向量
            current_features = self._extract_frame_features(frame_id)
            
            # 将当前帧特征添加到时序缓冲区
            self.temporal_buffer.append(current_features)
            
            # 如果缓冲区未满，返回None
            if len(self.temporal_buffer) < self.time_window_frames:
                return None
            
            # 将时序数据转换为n×T矩阵
            temporal_matrix = np.array(list(self.temporal_buffer)).T  # 转置为n×T
            
            return temporal_matrix
            
        except Exception as e:
            print(f"提取时序特征时出错: {e}")
            return None
    
    def _extract_frame_features(self, frame_id):
        """
        提取单帧的所有传感器特征
        
        参数:
            frame_id: 帧ID
            
        返回:
            list: 包含所有传感器特征的列表
        """
        frame_data = self.proprio_data[frame_id]
        features = []
        
        try:
            # 1. 提取IMU数据（原始值，不做带通滤波）
            imu_features = self._extract_imu_raw_data(frame_data)
            features.extend(imu_features)
            
            # 2. 提取底盘数据
            chassis_features = self._extract_chassis_data(frame_data)
            features.extend(chassis_features)
            
            return features
            
        except Exception as e:
            print(f"提取帧特征时出错: {e}")
            # 返回默认特征向量
            return [0.0] * (9 + 9)  # IMU(9) + 底盘(9)
    
    def _extract_imu_raw_data(self, frame_data):
        """
        提取IMU原始数据（不做带通滤波）
        
        参数:
            frame_data: 帧数据
            
        返回:
            list: 包含IMU原始特征的列表
        """
        try:
            imu_data = frame_data.get('imu', {})
            imu_features = []
            
            # 线性加速度（三轴）
            imu_features.append(imu_data.get('linear_acceleration_x', 0.0))
            imu_features.append(imu_data.get('linear_acceleration_y', 0.0) / 10)
            imu_features.append((imu_data.get('linear_acceleration_z', 0.0) - 9.8) / 10)
            
            # 角速度（三轴）
            imu_features.append(imu_data.get('angular_velocity_x', 0.0))
            imu_features.append(imu_data.get('angular_velocity_y', 0.0))
            imu_features.append(imu_data.get('angular_velocity_z', 0.0))
            
            # 姿态（三轴）
            imu_features.append(imu_data.get('orientation_x', 0.0))
            imu_features.append(imu_data.get('orientation_y', 0.0))
            imu_features.append(imu_data.get('orientation_z', 0.0))
            
            return imu_features
            
        except Exception as e:
            print(f"提取IMU原始数据时出错: {e}")
            return [0.0] * 9
    
    def _extract_chassis_data(self, frame_data):
        """
        从帧数据中提取底盘信息
        
        参数:
            frame_data (dict): 帧数据
            
        返回:
            list: 包含底盘信息的列表
        """
        try:
            chassis_data = []
            chassis = frame_data.get('chassis', {})
            
            # 车轮速度（四个车轮）
            chassis_data.append(chassis.get('lf_wheel_speed', 0.0) / 3.6 / 30)
            chassis_data.append(chassis.get('rf_wheel_speed', 0.0) / 3.6 / 30)
            chassis_data.append(chassis.get('lr_wheel_speed', 0.0) / 3.6 / 30)
            chassis_data.append(chassis.get('rr_wheel_speed', 0.0) / 3.6 / 30)
            
            # 当前速度
            chassis_data.append(chassis.get('cur_spd', 0.0) / 30)
            
            # 转向角
            chassis_data.append(chassis.get('cur_steer_wheel_angle', 0.0) / 30)
            
            # 燃油
            chassis_data.append(chassis.get('cur_fuel', 0.0) / 100)
            
            # 刹车
            chassis_data.append(chassis.get('cur_brake', 0.0) / 100)
            
            # 发动机转速
            chassis_data.append(chassis.get('cur_engine_speed', 0.0) / 8031.875)
            
            return chassis_data
            
        except Exception as e:
            print(f"提取底盘数据时出错: {e}")
            return [0.0] * 9
    
    def _get_label(self, label):
        """
        根据标签名称获取类别标签
        
        参数:
            label: 标签名称
            
        返回:
            int: 类别标签 (0-3)
        """
        try:
            return self.classes[label]
            
        except Exception as e:
            print(f"获取类别标签时出错: {e}")
            return None
    
    def _prepare_samples(self):
        """准备时序训练样本"""
        print("准备时序训练样本...")
        
        valid_samples = 0
        
        # 按时间戳排序帧ID
        sorted_frame_ids = sorted(self.proprio_data.keys(), key=lambda x: int(x))
        
        for label, tss in self.roi_data.items():
            start_ts = tss['start_ts']
            end_ts = tss['end_ts']
            
            # 清空时序缓冲区
            self.temporal_buffer.clear()
            
            for frame_id in sorted_frame_ids:
                ts = int(frame_id)
                if ts < start_ts or ts > end_ts:
                    continue
                
                # 提取时序特征
                temporal_features = self._extract_temporal_features(frame_id)
                
                # 如果时序特征未准备好（缓冲区未满），跳过
                if temporal_features is None:
                    continue
                
                # 获取类别标签
                label_id = self._get_label(label)
                if label_id is None:
                    continue
                
                # 添加到样本列表
                self.samples.append({
                    'temporal_features': temporal_features,
                    'label': label_id,
                    'frame_id': frame_id
                })
                
                valid_samples += 1
        
        print(f"准备完成 {valid_samples} 个时序训练样本")
        if valid_samples > 0:
            sample_shape = self.samples[0]['temporal_features'].shape
            print(f"时序特征维度: {sample_shape} (特征数×时序长度)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # 时序特征形状为 (n_features, time_steps)
        temporal_features = torch.tensor(sample['temporal_features'], dtype=torch.float32)
        label = torch.tensor(sample['label'], dtype=torch.long)
        return temporal_features, label
    
    def get_feature_dimensions(self):
        """获取特征维度信息"""
        if len(self.samples) == 0:
            return None, None
        
        sample = self.samples[0]
        temporal_features = sample['temporal_features']
        n_features, time_steps = temporal_features.shape
        
        return n_features, time_steps


# 测试函数
def test_temporal_dataset():
    """测试时序数据集"""
    try:
        # 使用一个测试数据集路径
        test_path = "/home/xie/xzk/Graduation/Terrain/dataset/dataset/2024-09-22-16-12-28"
        
        print("创建时序数据集...")
        dataset = TemporalTerrainDataset(test_path, time_window_seconds=1.0, sampling_rate=50)
        
        print(f"数据集大小: {len(dataset)}")
        
        if len(dataset) > 0:
            # 获取第一个样本
            features, label = dataset[0]
            print(f"时序特征形状: {features.shape}")
            print(f"标签: {label}")
            
            # 获取特征维度
            n_features, time_steps = dataset.get_feature_dimensions()
            print(f"特征数量: {n_features}")
            print(f"时序长度: {time_steps}")
        
        return dataset
        
    except Exception as e:
        print(f"测试时序数据集时出错: {e}")
        return None


if __name__ == "__main__":
    test_temporal_dataset()
