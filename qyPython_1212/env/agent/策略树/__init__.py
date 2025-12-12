"""
策略树模块 - 基于行为树的无人机战斗AI

模块结构：
- bt_framework.py: 行为树框架（BTNode, Sequence, Selector等）
- fire_control.py: 智能火控系统（SmartFireControl）
- agent.py: 主智能体类（BTDemoAgent）
- actions/: 动作节点
  - basic_actions.py: 基础动作
  - evasion.py: 导弹规避
  - attack.py: 攻击逻辑
  - formation.py: 阵型战术
"""

# 导出主要类，保持向后兼容
from .bt_framework import (
    BTNode,
    NodeStatus,
    Sequence,
    Selector,
    Action,
    Condition
)

from .fire_control import SmartFireControl

from .agent import BTDemoAgent

from .actions import (
    ActionResetFrame,
    ConditionCheckInitialDeployment,
    ActionExecuteDeployment,
    ActionEvadeMissiles,
    ActionAttackLogic,
    ActionSearchFormation,
    ActionMannedRetreat,
    ActionProtectMannedVision,
    ActionCenterPatrol,
    ActionPatrolFormation
)

__all__ = [
    # 框架
    'BTNode',
    'NodeStatus',
    'Sequence',
    'Selector',
    'Action',
    'Condition',
    # 火控
    'SmartFireControl',
    # 智能体
    'BTDemoAgent',
    # 动作节点
    'ActionResetFrame',
    'ConditionCheckInitialDeployment',
    'ActionExecuteDeployment',
    'ActionEvadeMissiles',
    'ActionAttackLogic',
    'ActionSearchFormation',
    'ActionMannedRetreat',
    'ActionProtectMannedVision',
    'ActionCenterPatrol',
    'ActionPatrolFormation',
]
