"""
动作节点模块

包含所有行为树的叶子节点（Action和Condition）

版本说明：
- V1（保守速度策略）：根据威胁等级使用不同速度（80%-100%）
- V2（当前版本）：所有威胁等级统一使用最大速度
- V3：新增单环/双环机动战术
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

# 机动战术（单环/双环）
from .circle_maneuver import (
    CircleManeuverTactics,
    ActionCircleManeuver,
    ActionAdaptiveManeuver
)

__all__ = [
    # 基础动作
    'ActionResetFrame',
    'ConditionCheckInitialDeployment',
    'ActionExecuteDeployment',
    # 规避
    'ActionEvadeMissiles',
    'ActionEvadeMissilesAdvanced',
    'ActionTacticalEvasion',  # 智能战术躲避（双机夹击检测+侧翼包抄）
    # 攻击
    'ActionAttackLogic',
    # 阵型
    'ActionSearchFormation',
    'ActionMannedRetreat',
    'ActionProtectMannedVision',
    'ActionCenterPatrol',
    'ActionPatrolFormation',
    # 机动战术
    'CircleManeuverTactics',      # 机动战术计算器
    'ActionCircleManeuver',       # 单环/双环机动动作节点
    'ActionAdaptiveManeuver',     # 自适应机动战术
]
