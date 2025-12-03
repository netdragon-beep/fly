# -*- coding: utf-8 -*-
"""
VEB-RL: Value-Evolutionary-Based Reinforcement Learning

úŽº‡: Value-Evolutionary-Based Reinforcement Learning (ICML 2024)

8ÃÄö:
- QNetwork: QQÜŒDueling QQÜ
- VEBIndividual: *S{QQÜ + îQÜ	
- VEBPopulation: Í¤¡
- VEBTrainer: ­Ãh+Elite Interaction, RL InjectionI	
- ReplayBuffer: ÏŒÞ>²:
"""

from .q_network import (
    QNetwork,
    DuelingQNetwork,
    encode_q_weights,
    decode_q_weights,
    create_target_network,
    soft_update,
    hard_update
)

from .veb_individual import (
    VEBIndividual,
    VEBPopulation,
    ReplayBuffer
)

from .veb_trainer import (
    VEBTrainer,
    GeneticOperators,
    MockEnvironment
)

from .reward import (
    RewardConfig,
    RewardCalculator
)

__all__ = [
    # QQÜ
    'QNetwork',
    'DuelingQNetwork',
    'encode_q_weights',
    'decode_q_weights',
    'create_target_network',
    'soft_update',
    'hard_update',
    # *SŒÍ¤
    'VEBIndividual',
    'VEBPopulation',
    'ReplayBuffer',
    # ­Ãh
    'VEBTrainer',
    'GeneticOperators',
    'MockEnvironment',
    # V±
    'RewardConfig',
    'RewardCalculator',
]

__version__ = '1.0.0'
