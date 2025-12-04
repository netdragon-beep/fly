# -*- coding: utf-8 -*-
"""
VEB-RL: Value-Evolutionary-Based Reinforcement Learning

: Value-Evolutionary-Based Reinforcement Learning (ICML 2024)

8:
- QNetwork: QQ܌Dueling QQ
- VEBIndividual: *S{QQ + Q	
- VEBPopulation: ͤ
- VEBTrainer: h+Elite Interaction, RL InjectionI	
- ReplayBuffer: ό>:
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
    # QQ
    'QNetwork',
    'DuelingQNetwork',
    'encode_q_weights',
    'decode_q_weights',
    'create_target_network',
    'soft_update',
    'hard_update',
    # *Sͤ
    'VEBIndividual',
    'VEBPopulation',
    'ReplayBuffer',
    # h
    'VEBTrainer',
    'GeneticOperators',
    'MockEnvironment',
    # V
    'RewardConfig',
    'RewardCalculator',
]

__version__ = '1.0.0'
