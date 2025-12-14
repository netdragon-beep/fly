"""
动作节点模块

包含所有行为树的叶子节点（Action和Condition）

版本说明：
- V1（保守速度策略）：根据威胁等级使用不同速度（80%-100%）
- V2（当前版本）：所有威胁等级统一使用最大速度
"""

from .basic_actions import (
    ActionResetFrame,
    ConditionCheckInitialDeployment,
    ActionExecuteDeployment
)

# 当前版本
from .evasion import ActionEvadeMissiles, ActionEvadeMissilesAdvanced, ActionTacticalEvasion

from .attack import ActionAttackLogic

from .formation import (
    ActionSearchFormation,
    ActionProtectMannedVision,
    ActionCenterPatrol,
    ActionPatrolFormation
)

__all__ = [
    # 基础动作
    'ActionResetFrame',
    'ConditionCheckInitialDeployment',
    'ActionExecuteDeployment',
    # 规避
    'ActionEvadeMissiles',
    'ActionEvadeMissilesAdvanced',
    'ActionTacticalEvasion',  # 新增：智能战术躲避（双机夹击检测+侧翼包抄）
    # 攻击
    'ActionAttackLogic',
    # 阵型
    'ActionSearchFormation',
    'ActionMannedRetreat',
    'ActionProtectMannedVision',
    'ActionCenterPatrol',
    'ActionPatrolFormation',
]
