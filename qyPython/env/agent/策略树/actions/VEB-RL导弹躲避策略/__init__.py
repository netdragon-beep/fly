"""
VEB-RL导弹躲避策略模块

基于 Value-Evolutionary-Based Reinforcement Learning (VEB-RL) 论文实现
用于空战中的导弹规避决策

核心创新:
1. 维护Q网络种群用于躲避策略
2. 使用负TD误差作为适应度
3. 连续动作空间的机动决策
"""

from .veb_evasion import VEBEvasion, VEBEvasionConfig

__all__ = ['VEBEvasion', 'VEBEvasionConfig']
