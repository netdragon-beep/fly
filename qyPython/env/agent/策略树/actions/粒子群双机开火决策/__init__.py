"""
双机协同开火决策模块

核心思路:
1. 随机采样大量双机攻击场景
2. 通过仿真获取命中/未命中结果
3. 训练分类器学习命中边界
4. 推理时直接用分类器判断是否应该开火

目录结构:
- models/  : 保存训练好的分类器模型
- data/    : 保存采集的训练数据
- logs/    : 训练日志

三种训练模式:
1. 离线采集+训练: 先采集数据，再训练模型
2. 在线学习: 边采集边训练，ε-greedy探索
3. 合成数据: 使用物理模型生成数据快速测试

使用方法:
1. 在线学习 (推荐): python run_online_learning.py --episodes 50
2. 离线采集: python run_data_collection.py --mode single
3. GPU训练: python train_with_simulation.py --gpu 0
4. 测试模型: python train_dual_fire.py --mode test
"""

from .feature_extractor import DualFireFeatureExtractor, DualFireFeatures
from .data_collector import DualFireDataCollector
from .classifier import DualFireClassifier
from .sim_data_collector import (
    SimulationDataCollector,
    AcceleratedTrainingController,
    select_fire_targets
)
from .data_collection_agent import DualFireDataCollectionAgent
from .online_learning_agent import OnlineDualFireAgent, OnlineConfig

__all__ = [
    # 核心组件
    'DualFireFeatureExtractor',
    'DualFireFeatures',
    'DualFireDataCollector',
    'DualFireClassifier',
    # 仿真数据采集
    'SimulationDataCollector',
    'AcceleratedTrainingController',
    'select_fire_targets',
    # 数据采集智能体
    'DualFireDataCollectionAgent',
    # 在线学习智能体
    'OnlineDualFireAgent',
    'OnlineConfig'
]
