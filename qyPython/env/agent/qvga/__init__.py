# Q-Value Based Genetic Algorithm (QVGA)
# 基于Q值的遗传算法训练框架

from .individual import Individual, PolicyNetwork, encode_weights, decode_weights
from .population import Population
from .q_network import QNetwork, SimpleQNetwork, QNetworkTrainer
from .genetic_operators import (
    tournament_selection,
    uniform_crossover,
    gaussian_mutation,
    create_next_generation
)
from .trainer import QVGATrainer, MockBattleEnvironment
from .qvga_agent import QVGAAutoAgent
from .reward import RewardCalculator, RewardConfig
from .real_battle_env import RealBattleEnvironment, QVGABattleAgent

__all__ = [
    # 个体编码
    'Individual',
    'PolicyNetwork',
    'encode_weights',
    'decode_weights',

    # 种群管理
    'Population',

    # Q网络
    'QNetwork',
    'SimpleQNetwork',
    'QNetworkTrainer',

    # 遗传算子
    'tournament_selection',
    'uniform_crossover',
    'gaussian_mutation',
    'create_next_generation',

    # 训练器
    'QVGATrainer',
    'MockBattleEnvironment',

    # Agent接口
    'QVGAAutoAgent',

    # 奖励函数
    'RewardCalculator',
    'RewardConfig',

    # 真实环境
    'RealBattleEnvironment',
    'QVGABattleAgent',
]
