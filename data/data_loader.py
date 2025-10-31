import json
import os

class Proprio_Loader:
    """
    数据加载器类，用于读取各种JSON数据文件
    """
    def __init__(self, dir):

        self.roi_frame_path = os.path.join(dir, 'roi_frame.json')
        self.proprio_infos_path = os.path.join(dir, 'proprio_infos.json')

        self.roi_frames = None
        self.proprio_infos = None
    
    def load_roi_frame_json(self):
        """
        读取 roi_frame.json 文件
        
        返回:
            dict: 包含 roi_frame 数据的字典
        """
        try:
            with open(self.roi_frame_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            print(f"成功加载 roi_frame.json，数据长度: {len(data)}")
            self.roi_frames = data
            return data
        except FileNotFoundError:
            print(f"错误: 文件 {self.roi_frame_path} 不存在")
            return {}
        except json.JSONDecodeError:
            print(f"错误: 文件 {self.roi_frame_path} 不是有效的JSON格式")
            return {}
        except Exception as e:
            print(f"读取文件时发生错误: {e}")
            return {}

    
    def load_proprio_seg_infos_json(self):
        """
        读取 proprio_seg_infos.json 文件
        
        返回:
            dict: 包含 proprio_seg_infos 数据的字典
        """
        try:
            with open(self.proprio_infos_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            print(f"成功加载 proprio_seg_infos.json，数据长度: {len(data)}")
            self.proprio_infos = data
            return data
        except FileNotFoundError:
            print(f"错误: 文件 {self.proprio_infos_path} 不存在")
            return {}
        except json.JSONDecodeError:
            print(f"错误: 文件 {self.proprio_infos_path} 不是有效的JSON格式")
            return {}
        except Exception as e:
            print(f"读取文件时发生错误: {e}")

# 使用示例
if __name__ == "__main__":
    # 测试加载 roi_frame.json
    dir = '/home/xie/xzk/Graduation/Terrain/dataset/dataset/2024-09-22-16-12-28'
    data_loader = Proprio_Loader(dir)
    data_loader.load_roi_frame_json()
    
    # 测试加载 proprio_seg_infos.json
    data_loader.load_proprio_seg_infos_json()
