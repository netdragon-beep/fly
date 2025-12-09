"""
行为树战斗AI - 向后兼容入口文件

注意：此文件为向后兼容保留。
新代码请直接从对应模块导入：
- from 策略树 import BTDemoAgent
- from 策略树.fire_control import SmartFireControl
- from 策略树.actions import ActionEvadeMissiles

模块结构：
策略树/
├── __init__.py          # 包入口
├── bt_framework.py      # 行为树框架
├── fire_control.py      # 智能火控系统
├── agent.py             # BTDemoAgent主类
└── actions/             # 动作节点
    ├── __init__.py
    ├── basic_actions.py # 基础动作
    ├── evasion.py       # 导弹规避
    ├── attack.py        # 攻击逻辑
    └── formation.py     # 阵型战术
"""

# 向后兼容：从新模块重新导出所有类
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
