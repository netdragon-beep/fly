"""
动作节点模块

包含所有行为树的叶子节点（Action和Condition）

版本说明：
- V1（保守速度策略）：根据威胁等级使用不同速度（80%-100%）
- V2（当前版本）：所有威胁等级统一使用最大速度
- V3：新增单环/双环机动战术
- V4：新增领航跟随约束（Leader-Follower）
"""

from .basic_actions import (
    ActionResetFrame,
    ActionMannedFollowConstraint,  # 新增：领航跟随约束
    ConditionCheckInitialDeployment,
    ActionExecuteDeployment
)

# 当前版本
from .evasion import ActionEvadeMissiles, ActionEvadeMissilesAdvanced, ActionTacticalEvasion

from .attack import ActionAttackLogic

from .formation import (
    ActionSearchFormation,
    ActionProtectMannedVision,
    ActionCenterPriority,      # 有人机中心优先
    ActionCenterPatrol,        # 无人机外围警戒
    ActionPatrolFormation
)

# 机动战术（单环/双环）
from .circle_maneuver import (
    CircleManeuverTactics,
    ActionCircleManeuver,
    ActionAdaptiveManeuver
)

# 咬尾机制
from .tail_chase import (
    TailChaseState,
    CooperationMode,
    TailChaseParams,
    TailChaseCoordinator,
    TailChaseTactics,
    ActionTailChase
)

__all__ = [
    # 基础动作
    'ActionResetFrame',
    'ActionMannedFollowConstraint',   # 领航跟随约束（强制）
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
    'ActionProtectMannedVision',
    'ActionCenterPriority',       # 有人机中心优先（占领中心区域）
    'ActionCenterPatrol',         # 无人机外围警戒（18km）
    'ActionPatrolFormation',
    # 机动战术
    'CircleManeuverTactics',      # 机动战术计算器
    'ActionCircleManeuver',       # 单环/双环机动动作节点
    'ActionAdaptiveManeuver',     # 自适应机动战术
    # 咬尾机制
    'TailChaseState',             # 咬尾状态枚举
    'CooperationMode',            # 协同模式枚举
    'TailChaseParams',            # 咬尾参数
    'TailChaseCoordinator',       # 协同决策器
    'TailChaseTactics',           # 战术计算器
    'ActionTailChase',            # 咬尾行为树节点
]
