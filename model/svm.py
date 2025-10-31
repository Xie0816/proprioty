import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


class SVMClassifier:
    """
    SVM分类器模型
    """
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42):
        """
        初始化SVM模型
        
        参数:
            kernel (str): 核函数类型 ('linear', 'poly', 'rbf', 'sigmoid')
            C (float): 正则化参数
            gamma (str or float): 核系数 ('scale', 'auto' or float)
            probability (bool): 是否启用概率预测
            random_state (int): 随机种子
        """
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=probability,
            random_state=random_state
        )
        self.is_fitted = False
        self.class_mapping = None
    
    def fit(self, X, y, class_mapping=None):
        """
        训练SVM模型
        
        参数:
            X (array-like): 训练特征
            y (array-like): 训练标签
            class_mapping (dict): 类别映射字典
        """
        # 保存类别映射
        if class_mapping is not None:
            self.class_mapping = class_mapping
        
        # 直接使用原始特征训练模型（不移除归一化）
        self.model.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        """
        预测类别
        
        参数:
            X (array-like): 输入特征
            
        返回:
            array: 预测的类别标签
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        预测类别概率
        
        参数:
            X (array-like): 输入特征
            
        返回:
            array: 各类别的概率
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        return self.model.predict_proba(X)
    
    def predict_with_details(self, X):
        """
        详细预测，返回预测结果和概率
        
        参数:
            X (array-like): 输入特征
            
        返回:
            dict: 包含预测结果的字典
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            result = {
                'predicted_label': pred,
                'probabilities': probs.tolist(),
                'confidence': probs[pred]
            }
            
            # 添加类别名称（如果存在类别映射）
            if self.class_mapping is not None:
                result['predicted_class'] = self.class_mapping.get(pred, f"类别_{pred}")
                
                # 创建概率字典
                prob_dict = {}
                for j, prob in enumerate(probs):
                    prob_name = self.class_mapping.get(j, f"类别_{j}")
                    prob_dict[prob_name] = prob
                result['probability_dict'] = prob_dict
            
            results.append(result)
        
        return results
    
    def evaluate(self, X, y, verbose=True):
        """
        评估模型性能
        
        参数:
            X (array-like): 测试特征
            y (array-like): 测试标签
            verbose (bool): 是否打印详细结果
            
        返回:
            dict: 评估结果
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        results = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'confusion_matrix': confusion_matrix(y, y_pred)
        }
        
        if verbose:
            print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # 打印分类报告
            if self.class_mapping is not None:
                class_names = [self.class_mapping.get(i, f"类别_{i}") for i in range(len(self.class_mapping))]
                print("\n详细分类报告:")
                print(classification_report(y, y_pred, target_names=class_names))
            
            # 打印混淆矩阵
            print("\n混淆矩阵:")
            cm = results['confusion_matrix']
            if self.class_mapping is not None:
                class_names = [self.class_mapping.get(i, f"类别_{i}") for i in range(len(self.class_mapping))]
                print("预测\\真实", end="")
                for name in class_names:
                    print(f"\t{name}", end="")
                print()
                
                for i, true_name in enumerate(class_names):
                    print(f"{true_name}", end="")
                    for j in range(len(class_names)):
                        print(f"\t{cm[i, j]}", end="")
                    print()
        
        return results
    
    def save_model(self, model_path, info_path):
        """
        保存模型和相关信息
        
        参数:
            model_path (str): 模型保存路径
            info_path (str): 模型信息保存路径
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，无法保存")
        
        # 保存SVM模型
        joblib.dump(self.model, model_path)
        
        # 保存模型信息
        model_info = {
            'class_mapping': self.class_mapping,
            'kernel': self.model.kernel,
            'C': self.model.C,
            'gamma': self.model.gamma,
            'is_fitted': self.is_fitted
        }
        
        import json
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
    
    def load_model(self, model_path, info_path):
        """
        加载模型和相关信息
        
        参数:
            model_path (str): 模型加载路径
            info_path (str): 模型信息加载路径
        """
        # 加载SVM模型
        self.model = joblib.load(model_path)
        
        # 加载模型信息
        import json
        with open(info_path, 'r') as f:
            model_info = json.load(f)
        
        self.class_mapping = model_info['class_mapping']
        self.is_fitted = model_info['is_fitted']
        
        return self


def create_svm_model(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42):
    """
    创建SVM模型的便捷函数
    
    参数:
        kernel (str): 核函数类型 ('linear', 'poly', 'rbf', 'sigmoid')
        C (float): 正则化参数
        gamma (str or float): 核系数 ('scale', 'auto' or float)
        probability (bool): 是否启用概率预测
        random_state (int): 随机种子
    
    返回:
        SVMClassifier: 初始化的SVM模型
    """
    model = SVMClassifier(
        kernel=kernel,
        C=C,
        gamma=gamma,
        probability=probability,
        random_state=random_state
    )
    return model


if __name__ == "__main__":
    # 测试模型
    print("测试SVM模型...")
    
    # 创建示例数据
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 4, 100)
    
    # 创建模型
    model = create_svm_model()
    
    # 训练模型
    class_mapping = {
        0: "类别_0",
        1: "类别_1", 
        2: "类别_2",
        3: "类别_3"
    }
    model.fit(X_train, y_train, class_mapping)
    
    # 测试预测
    X_test = np.random.randn(5, 10)
    predictions = model.predict(X_test)
    detailed_predictions = model.predict_with_details(X_test)
    
    print(f"输入形状: {X_test.shape}")
    print(f"预测结果: {predictions}")
    print(f"详细预测: {detailed_predictions[0]}")
    
    # 测试评估
    y_test = np.random.randint(0, 4, 5)
    model.evaluate(X_test, y_test)
    
    print("模型测试完成！")
