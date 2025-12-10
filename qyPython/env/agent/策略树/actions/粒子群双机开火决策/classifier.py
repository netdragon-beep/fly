"""
双机协同开火分类器

基于采集的数据训练分类器
用于判断当前状态下双机协同开火是否能命中

支持多种分类器:
1. MLP (多层感知机) - PyTorch实现
2. XGBoost - 梯度提升树
3. RandomForest - 随机森林
"""

import os
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    from .feature_extractor import DualFireFeatureExtractor, DualFireFeatures
except ImportError:
    from feature_extractor import DualFireFeatureExtractor, DualFireFeatures


# ==================== MLP分类器 ====================

class MLPClassifier(nn.Module):
    """多层感知机分类器"""

    def __init__(self, input_dim: int = 16, hidden_dims: Tuple[int, ...] = (128, 64, 32)):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        # 输出层: 命中概率
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ==================== 双机开火分类器 ====================

class DualFireClassifier:
    """
    双机协同开火分类器

    核心功能:
    1. 训练: 从采集的数据学习命中边界
    2. 推理: 判断当前状态是否应该开火
    """

    def __init__(self, model_type: str = 'mlp', device: str = None):
        """
        Args:
            model_type: 'mlp', 'xgboost', 'random_forest'
            device: 'cuda' 或 'cpu'
        """
        self.model_type = model_type
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.feature_extractor = DualFireFeatureExtractor()
        self.is_trained = False

        # 训练统计
        self.train_history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 100,
              batch_size: int = 64,
              lr: float = 1e-3,
              val_split: float = 0.2,
              verbose: bool = True) -> Dict:
        """
        训练分类器

        Args:
            X: 特征矩阵 (N, feature_dim)
            y: 标签 (N,)
            epochs: 训练轮数
            batch_size: 批大小
            lr: 学习率
            val_split: 验证集比例
            verbose: 是否打印进度

        Returns:
            训练历史
        """
        if verbose:
            print(f"开始训练 {self.model_type} 分类器...")
            print(f"  数据: {X.shape[0]} 样本, {X.shape[1]} 特征")

        if self.model_type == 'mlp':
            return self._train_mlp(X, y, epochs, batch_size, lr, val_split, verbose)
        elif self.model_type == 'xgboost':
            return self._train_xgboost(X, y, verbose)
        elif self.model_type == 'random_forest':
            return self._train_random_forest(X, y, verbose)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def _train_mlp(self, X: np.ndarray, y: np.ndarray,
                   epochs: int, batch_size: int, lr: float,
                   val_split: float, verbose: bool) -> Dict:
        """训练MLP分类器"""
        # 划分训练集和验证集
        n_samples = X.shape[0]
        n_val = int(n_samples * val_split)
        indices = np.random.permutation(n_samples)

        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # 转为PyTorch张量
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)

        # 创建数据加载器
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 创建模型
        self.model = MLPClassifier(input_dim=X.shape[1]).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        # 训练循环
        best_val_acc = 0
        best_model_state = None

        for epoch in range(epochs):
            # 训练
            self.model.train()
            train_loss = 0
            train_correct = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_X.size(0)
                pred = (output > 0.5).float()
                train_correct += (pred == batch_y).sum().item()

            train_loss /= len(train_idx)
            train_acc = train_correct / len(train_idx)

            # 验证
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(X_val_t)
                val_loss = criterion(val_output, y_val_t).item()
                val_pred = (val_output > 0.5).float()
                val_acc = (val_pred == y_val_t).sum().item() / len(val_idx)

            # 记录历史
            self.train_history['loss'].append(train_loss)
            self.train_history['accuracy'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_accuracy'].append(val_acc)

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: "
                      f"loss={train_loss:.4f}, acc={train_acc:.4f}, "
                      f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        # 恢复最佳模型
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        self.is_trained = True

        if verbose:
            print(f"训练完成! 最佳验证准确率: {best_val_acc:.4f}")

        return self.train_history

    def _train_xgboost(self, X: np.ndarray, y: np.ndarray, verbose: bool) -> Dict:
        """训练XGBoost分类器"""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("请安装xgboost: pip install xgboost")

        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss'
        )

        self.model.fit(X, y, verbose=verbose)
        self.is_trained = True

        # 特征重要性
        if verbose:
            importance = self.model.feature_importances_
            feature_names = DualFireFeatures.feature_names()
            print("\n特征重要性:")
            for name, imp in sorted(zip(feature_names, importance), key=lambda x: -x[1])[:5]:
                print(f"  {name}: {imp:.4f}")

        return {'feature_importance': self.model.feature_importances_.tolist()}

    def _train_random_forest(self, X: np.ndarray, y: np.ndarray, verbose: bool) -> Dict:
        """训练随机森林分类器"""
        from sklearn.ensemble import RandomForestClassifier

        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            n_jobs=-1
        )

        self.model.fit(X, y)
        self.is_trained = True

        if verbose:
            print(f"训练完成!")

        return {'feature_importance': self.model.feature_importances_.tolist()}

    def predict(self, shooter1: Dict, shooter2: Dict, target: Dict) -> Tuple[bool, float, str]:
        """
        预测是否应该开火

        Args:
            shooter1, shooter2: 射手信息
            target: 目标信息

        Returns:
            (should_fire, confidence, reason)
        """
        if not self.is_trained:
            raise RuntimeError("模型未训练")

        # 提取特征
        features = self.feature_extractor.extract(shooter1, shooter2, target)
        X = features.to_array().reshape(1, -1)

        # 预测
        if self.model_type == 'mlp':
            self.model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X).to(self.device)
                prob = self.model(X_t).item()
        else:
            prob = self.model.predict_proba(X)[0, 1]

        # 决策
        should_fire = prob > 0.5
        confidence = prob if should_fire else 1 - prob

        # 生成原因说明
        reason = self._generate_reason(features, prob, should_fire)

        return should_fire, confidence, reason

    def predict_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """批量预测"""
        if not self.is_trained:
            raise RuntimeError("模型未训练")

        if self.model_type == 'mlp':
            self.model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X).to(self.device)
                probs = self.model(X_t).cpu().numpy().flatten()
        else:
            probs = self.model.predict_proba(X)[:, 1]

        predictions = (probs > 0.5).astype(int)
        return predictions, probs

    def _generate_reason(self, features: DualFireFeatures, prob: float, should_fire: bool) -> str:
        """生成决策原因说明"""
        reasons = []

        # 角度差
        if features.angle_diff >= 60:
            reasons.append(f"角度差大({features.angle_diff:.0f}°)")
        elif features.angle_diff < 30:
            reasons.append(f"角度差小({features.angle_diff:.0f}°)")

        # 距离
        avg_dist = (features.dist_1 + features.dist_2) / 2
        if avg_dist < 15000:
            reasons.append(f"距离近({avg_dist/1000:.1f}km)")
        elif avg_dist > 25000:
            reasons.append(f"距离远({avg_dist/1000:.1f}km)")

        # 目标状态 (现在分别判断两个射手看到的横向速度)
        max_lateral = max(features.target_v_lateral_1, features.target_v_lateral_2)
        min_lateral = min(features.target_v_lateral_1, features.target_v_lateral_2)
        lateral_diff = max_lateral - min_lateral

        if max_lateral > 200:
            reasons.append(f"目标横向速度大({max_lateral:.0f}m/s)")

        # 协同优势判断
        if lateral_diff > 100:
            reasons.append(f"横向速度差异大({lateral_diff:.0f}m/s,难以同时躲避)")

        if features.target_maneuver > 0.5:
            reasons.append("目标在机动")

        if should_fire:
            return f"建议开火(概率{prob:.0%}): " + ", ".join(reasons) if reasons else f"建议开火(概率{prob:.0%})"
        else:
            return f"不建议开火(概率{1-prob:.0%}): " + ", ".join(reasons) if reasons else f"不建议开火(概率{1-prob:.0%})"

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """评估模型性能"""
        predictions, probs = self.predict_batch(X)

        accuracy = (predictions == y).mean()
        precision = (predictions[y == 1] == 1).mean() if y.sum() > 0 else 0
        recall = (y[predictions == 1] == 1).mean() if predictions.sum() > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def save(self, filepath: str):
        """保存模型"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if self.model_type == 'mlp':
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_type': self.model_type,
                'train_history': self.train_history
            }, filepath)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'model_type': self.model_type,
                    'train_history': self.train_history
                }, f)

        print(f"模型已保存: {filepath}")

    def load(self, filepath: str):
        """加载模型"""
        if self.model_type == 'mlp':
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            self.model = MLPClassifier().to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.train_history = checkpoint.get('train_history', {})
        else:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.model = data['model']
            self.train_history = data.get('train_history', {})

        self.is_trained = True
        print(f"模型已加载: {filepath}")


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("双机协同开火分类器测试")
    print("=" * 60)

    # 生成测试数据
    try:
        from .data_collector import DualFireDataCollector
    except ImportError:
        from data_collector import DualFireDataCollector

    collector = DualFireDataCollector()
    collector.generate_synthetic_samples(num_samples=5000, verbose=True)
    X, y = collector.get_training_data()

    # 训练MLP分类器
    classifier = DualFireClassifier(model_type='mlp')
    classifier.train(X, y, epochs=50, verbose=True)

    # 评估
    metrics = classifier.evaluate(X, y)
    print(f"\n评估结果:")
    print(f"  准确率: {metrics['accuracy']:.4f}")
    print(f"  精确率: {metrics['precision']:.4f}")
    print(f"  召回率: {metrics['recall']:.4f}")
    print(f"  F1分数: {metrics['f1']:.4f}")

    # 测试预测
    print("\n测试预测:")
    shooter1 = {
        'longitude': 145.9, 'latitude': 33.5, 'altitude': 5000,
        'speed': 400, 'heading': 0.5,
        'velocity_x': 200, 'velocity_y': 346, 'velocity_z': 0
    }
    shooter2 = {
        'longitude': 146.1, 'latitude': 33.3, 'altitude': 5500,
        'speed': 400, 'heading': 2.5,
        'velocity_x': -200, 'velocity_y': 346, 'velocity_z': 0
    }
    target = {
        'longitude': 146.0, 'latitude': 33.4, 'altitude': 5200,
        'speed': 350, 'heading': 1.0,
        'v_x': 100, 'v_y': -300, 'v_z': 0,
        'roll': 0.5, 'platform_entity_type': '无人机'
    }

    should_fire, confidence, reason = classifier.predict(shooter1, shooter2, target)
    print(f"  决策: {'开火' if should_fire else '等待'}")
    print(f"  置信度: {confidence:.2%}")
    print(f"  原因: {reason}")
