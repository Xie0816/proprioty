import numpy as np
import torch
from scipy import signal
from scipy.integrate import simps
from data.data_loader import Proprio_Loader
from torch.utils.data import Dataset
from collections import deque


class TerrainDataset(Dataset):
    """
    地形分类数据集类
    处理单个数据集文件夹，提取z轴加速度频域特征和速度信息作为输入
    输出为四类道路分类：Dirt_Road、Asphalt_Road、Cement_Road、Gravel_Road
    """
    
    def __init__(self, dataset_path):
        """初始化数据集"""
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
        
        self.bp_f = 50 #采样帧
        self.bp_sf = 50 # 采样频率
        self.his_tss = deque()
        
        # 加载数据并准备样本
        self._load_data()
        self._prepare_samples()
    
    def _load_data(self):
        """加载数据"""
        print(f"加载数据集: {self.dataset_path}")
        
        self.roi_data = self.loader.load_roi_frame_json()
        self.proprio_data = self.loader.load_proprio_seg_infos_json()
        
        if not self.roi_data or not self.proprio_data:
            raise ValueError("数据加载失败")
        
        print(f"成功加载数据:")
        print(f"  - roi_frame: {len(self.roi_data)} 个类别")
        print(f"  - proprio_seg_infos: {len(self.proprio_data)} 个帧")
    
    def _extract_features(self, frame_id):
        """
        从帧数据中提取特征
        输入：z轴加速度频域特征 + 速度信息
        
        参数:
            frame_id: 帧ID
            
        返回:
            numpy.array: 特征向量
        """
        frame_data = self.proprio_data[frame_id]

        features = []
        
        try:
            # 1. 提取imu数据
            imu_data = self._extract_imu_data(frame_id)
            features.extend(imu_data)
            # 2. 提取chassis信息
            chassis_data = self._extract_chassis_data(frame_data)
            features.extend(chassis_data)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"提取特征时出错: {e}")
            return None
    
    def _extract_imu_data(self, frame_id):
        """
        从帧数据中提取前self.bp_f帧的三轴加速度数据并计算bandpower
        
        参数:
            frame_id: 当前帧ID
            
        返回:
            list: 包含三轴加速度bandpower特征的列表
        """
        try:
            # 获取前self.bp_f帧的三轴加速度数据
            x_accelerations = []
            y_accelerations = []
            z_accelerations = []
            
            # 遍历前self.bp_f帧
            for frame_id in self.his_tss:
                # 获取前帧数据
                prev_frame_data = self.proprio_data.get(frame_id)
                if prev_frame_data is None:
                    continue
                
                # 提取加速度数据
                imu_data = prev_frame_data.get('imu', {})
                if 'linear_acceleration_x' in imu_data:
                    x_accelerations.append(imu_data['linear_acceleration_x'] / 10)
                if 'linear_acceleration_y' in imu_data:
                    y_accelerations.append(imu_data['linear_acceleration_y'] / 10)
                if 'linear_acceleration_z' in imu_data:
                    z_accelerations.append((imu_data['linear_acceleration_z'] - 9.8) / 10)
            
            features = []
            
            # 计算频带功率 (根据50Hz采样频率调整频带范围)
            bands = {
                'delta': [1, 8],
                'theta': [8, 15],
                'alpha': [15, 21],
                'beta': [21, 30]
            }
            
            # 为每个轴计算bandpower
            for axis_name, accelerations in [('x', x_accelerations), ('y', y_accelerations), ('z', z_accelerations)]:
                for band_name, band_range in bands.items():
                    try:
                        bp = self.bandpower(accelerations, band_range)
                        features.append(bp)
                    except Exception as e:
                        print(f"计算{axis_name}轴{band_name}频带功率时出错: {e}")
                        features.append(0.0)

            
            return features
            
        except Exception as e:
            print(f"提取IMU数据时出错: {e}")
            # 返回默认值 (3轴 * 4频带 = 12个特征)
            return [0.0] * 12
    
    def bandpower(self, data, band, window_sec=None, relative=False):
        """
        计算信号的频带功率
        
        参数:
            data (array): 输入信号
            bp_sf (float): 采样频率
            band (list): 频带范围 [low, high]
            window_sec (float): 窗口长度（秒）
            relative (bool): 是否返回相对功率
        
        返回:
            float: 频带功率
        """
        # 检查数据是否有效
        if len(data) < 2:
            return 0.0
        
        # 检查数据是否全为零或常数
        if np.all(data == data[0]):
            return 0.0
        
        try:
            # 计算功率谱密度
            if window_sec is not None:
                nperseg = window_sec * self.bp_sf
            else:
                nperseg = min(256, len(data))
            
            # 确保nperseg不超过数据长度
            nperseg = min(nperseg, len(data))
            
            freqs, psd = signal.welch(data, self.bp_sf, nperseg=nperseg)
            
            # 找到目标频带
            idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
            
            # 检查是否有有效的频带数据
            if not np.any(idx_band):
                return 0.0
            
            # 计算频带功率
            bp = simps(psd[idx_band], freqs[idx_band])
            
            if relative:
                bp_total = simps(psd, freqs)
                if bp_total > 0:
                    bp = bp / bp_total
                else:
                    bp = 0.0
            
            return bp
            
        except Exception as e:
            print(f"计算bandpower时出错: {e}, 数据长度: {len(data)}")
            return 0.0
    
    def _extract_chassis_data(self, frame_data):
        """
        从帧数据中提取速度信息
        
        参数:
            frame_data (dict): 帧数据
            
        返回:
            dict: 包含速度信息的字典
        """
        try:
            chassis_data = []
            chassis_data.append(frame_data['chassis']['lf_wheel_speed']/3.6/ 30) 
            chassis_data.append(frame_data['chassis']['rf_wheel_speed']/3.6 / 30) 
            chassis_data.append(frame_data['chassis']['lr_wheel_speed']/3.6 / 30) 
            chassis_data.append(frame_data['chassis']['rr_wheel_speed']/3.6 / 30) 
            chassis_data.append(frame_data['chassis']['cur_spd']   / 30) 
            chassis_data.append(frame_data['chassis']['cur_steer_wheel_angle'] / 30) 
            chassis_data.append(frame_data['chassis']['cur_fuel']  /100) 
            chassis_data.append(frame_data['chassis']['cur_brake']/100) 
            chassis_data.append(frame_data['chassis']['cur_engine_speed'] / 8031.875) 
            
            return chassis_data
            
        except Exception as e:
            print(f"提取速度数据时出错: {e}")
            return None
    
    def _get_label(self, label):
        """
        根据帧ID获取类别标签
        
        参数:
            frame_id: 帧ID
            
        返回:
            int: 类别标签 (0-3)
        """
        try:
            return self.classes[label]
            
        except Exception as e:
            print(f"获取类别标签时出错: {e}")
   
    def _prepare_samples(self):
        """准备训练样本"""
        print("准备训练样本...")
        
        valid_samples = 0
        
        for label, tss in self.roi_data.items():
            start_ts = tss['start_ts']
            end_ts = tss['end_ts']

            for frame_id in self.proprio_data.keys():
                ts = int(frame_id)
                if ts < start_ts or ts > end_ts:
                    continue

                        # 历史帧
                self.his_tss.append(frame_id)
                if len(self.his_tss) > self.bp_f:
                    self.his_tss.popleft()
                else:
                    continue
                # 提取特征
                
                features = self._extract_features(frame_id)
                
                # 获取类别标签
                label_id = self._get_label(label)
                
                # 添加到样本列表
                self.samples.append({
                    'features': features,
                    'label': label_id,
                    'frame_id': frame_id
                })
                
                valid_samples += 1
        
        print(f"准备完成 {valid_samples} 个训练样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = torch.tensor(sample['features'], dtype=torch.float32)
        label = torch.tensor(sample['label'], dtype=torch.long)
        return features, label
