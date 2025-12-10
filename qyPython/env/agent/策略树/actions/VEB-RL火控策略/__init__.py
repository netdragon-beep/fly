"""
VEB-RL火控策略模块

基于 Value-Evolutionary-Based Reinforcement Learning (VEB-RL) 论文
结合进化算法(EA)和基于价值的强化学习(Value-based RL)

核心创新:
1. 维护Q网络种群而非策略网络种群
2. 使用负TD误差作为适应度指标
3. 精英交互机制(Elite Interaction)
"""

from .veb_fire_control import VEBFireControl, VEBConfig
from .q_network import QNetwork, DuelingQNetwork

__all__ = ['VEBFireControl', 'VEBConfig', 'QNetwork', 'DuelingQNetwork']
