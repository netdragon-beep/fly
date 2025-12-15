"""
咬尾机制模块 (Tail Chase Mechanism)

核心概念：
=========
咬尾机动是一种近距离缠斗战术，通过绕飞进入敌机后半球（尾追位置），
利用敌机雷达盲区和反击困难的优势，在最优位置发起攻击。

工作原理：
=========
1. 搜索阶段(SEARCH): 寻找适合咬尾的目标
2. 接近阶段(APPROACH): 调整角度进入敌机后半球
3. 追踪阶段(CHASE): 绕飞咬尾，逐步缩小半径
4. 攻击阶段(ATTACK): 稳定跟踪并开火
5. 脱离阶段(DISENGAGE): 重新寻找目标

优势：
=====
- 敌机后方是雷达探测盲区（方位范围±60°，后方120°是盲区）
- 尾追时敌机只能横向机动，逃逸困难
- 尾追NEZ最小但命中率最高

官方参数（基于竞赛文档）：
========================
- 有人机：速度180-500m/s，最大径向加速度15m/s²，雷达60km/±60°
- 无人机：速度120-360m/s，最大径向加速度15m/s²，雷达40km/±60°
- 导弹：速度1200m/s，最大飞行时间60s，最大距离72km，杀伤半径10m
"""

import math
from enum import Enum
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from ..bt_framework import Action, NodeStatus
from utilities.yxScriptTreeFunc import YxScriptTreeFunc as decCmd
from utilities.yxGeoUtils import YxGeoUtils


class TailChaseState(Enum):
    """咬尾机制状态"""
    SEARCH = "search"           # 搜索目标
    APPROACH = "approach"       # 接近阶段
    CHASE = "chase"             # 追踪阶段
    ATTACK = "attack"           # 攻击阶段
    DISENGAGE = "disengage"     # 脱离阶段


class CooperationMode(Enum):
    """协同模式"""
    DUAL_ATTACK = "dual_attack"      # 双机协同夹击
    WAIT_SUPPORT = "wait_support"    # 等待支援
    SOLO_ATTACK = "solo_attack"      # 单机作战（限制条件）
    RETREAT = "retreat"              # 撤退回大部队
    NO_ACTION = "no_action"          # 不执行咬尾


@dataclass
class TailChaseParams:
    """咬尾机制参数（基于官方竞赛文档）"""

    # ========== 飞机性能参数（官方） ==========
    MANNED_MIN_SPEED: float = 180       # 有人机最小速度 m/s
    MANNED_MAX_SPEED: float = 500       # 有人机最大速度 m/s
    UAV_MIN_SPEED: float = 120          # 无人机最小速度 m/s
    UAV_MAX_SPEED: float = 360          # 无人机最大速度 m/s
    MAX_RADIAL_ACC: float = 15          # 最大径向加速度 m/s²

    # ========== 雷达参数（官方） ==========
    MANNED_RADAR_RANGE: float = 60000   # 有人机雷达距离 60km
    UAV_RADAR_RANGE: float = 40000      # 无人机雷达距离 40km
    RADAR_AZIMUTH: float = 60           # 雷达方位角 ±60°
    RADAR_BLIND_ZONE: float = 120       # 后方盲区 120°

    # ========== 高度限制（官方） ==========
    MIN_ALTITUDE: float = 2000          # 最低飞行高度 2000m
    MAX_ALTITUDE: float = 7000          # 最高飞行高度 7000m

    # ========== 阶段转换距离阈值 ==========
    SEARCH_RANGE: float = 25000         # 搜索范围 25km
    APPROACH_RANGE: float = 15000       # 接近范围 15km
    CHASE_RANGE: float = 10000          # 追踪范围 10km

    # ========== 姿态角阈值（度） ==========
    # 姿态角定义：0°=迎头，90°=横越，180°=尾追
    APPROACH_ANGLE: float = 90          # 进入后半球
    CHASE_ANGLE: float = 120            # 稳定咬尾
    ATTACK_ANGLE: float = 135           # 最优攻击角度

    # ========== NEZ参数（基于官方导弹参数估算） ==========
    # 导弹：1200m/s，60s最大飞行时间，72km最大距离
    MANNED_HEAD_ON_NEZ: float = 20000   # 有人机迎头NEZ 20km
    MANNED_TAIL_NEZ: float = 8000       # 有人机尾追NEZ 8km
    UAV_HEAD_ON_NEZ: float = 18000      # 无人机迎头NEZ 18km
    UAV_TAIL_NEZ: float = 6000          # 无人机尾追NEZ 6km

    # ========== 绕飞半径 ==========
    # 基于转弯半径公式 R = V²/a
    # 有人机350m/s: 8.2km, 无人机250m/s: 4.2km
    RADIUS_APPROACH: float = 12000      # 接近阶段半径 12km
    RADIUS_CHASE: float = 10000         # 追踪阶段半径 10km
    RADIUS_ATTACK: float = 8000         # 攻击阶段半径 8km
    RADIUS_MIN: float = 6000            # 最小半径 6km

    # ========== 半径调整因子 ==========
    RADIUS_SHRINK_RATE: float = 0.95    # 每次缩小5%
    RADIUS_EXPAND_RATE: float = 1.05    # 每次扩大5%

    # ========== 速度策略 ==========
    # 最佳转弯速度（能量机动理论）
    CORNER_SPEED_MANNED: float = 350    # 有人机最佳转弯速度 m/s
    CORNER_SPEED_UAV: float = 250       # 无人机最佳转弯速度 m/s
    CHASE_SPEED_RATIO: float = 0.9      # 追踪时使用90%最大速度
    ATTACK_SPEED_RATIO: float = 0.85    # 攻击时稳定速度

    # ========== 高度优势 ==========
    ALTITUDE_ADVANTAGE: float = 300     # 保持比敌机高300m

    # ========== 超时参数（帧数，10帧/秒） ==========
    CHASE_TIMEOUT: int = 600            # 追踪超时600帧（60秒）
    APPROACH_TIMEOUT: int = 300         # 接近超时300帧（30秒）
    DISENGAGE_DURATION: int = 100       # 脱离持续100帧（10秒）

    # ========== 协同参数 ==========
    TEAMMATE_NEARBY_DISTANCE: float = 15000  # 友机"附近"定义 15km
    MAIN_FORCE_DISTANCE: float = 30000       # 大部队距离 30km
    MIN_UNITS_FOR_SOLO: int = 4              # 允许单挑的最小我方数量

    # ========== 开火条件 ==========
    PK_THRESHOLD_DUAL: float = 0.35     # 双机协同开火阈值
    PK_THRESHOLD_SOLO: float = 0.50     # 单挑开火阈值（更严格）
    PK_THRESHOLD_DEFAULT: float = 0.45  # 默认开火阈值


class TailChaseCoordinator:
    """
    咬尾协同决策器

    核心原则：
    1. 优先双机协同：友机在附近时必须双机夹击
    2. 智能等待：无友机时不贸然进攻，拉开距离等待支援
    3. 限制单挑：只有特定条件才允许单机作战
    4. 数量保护：我方≤3架时禁止单机作战
    """

    def __init__(self, params: TailChaseParams = None):
        self.params = params or TailChaseParams()
        self._solo_attack_unit = None  # 当前执行单挑的单位名称

    def decide_cooperation_mode(
        self,
        unit: Dict,
        target: Dict,
        own_units: List[Dict],
        enemy_units: List[Dict],
        tail_chase_states: Dict = None
    ) -> Tuple[CooperationMode, Optional[Dict]]:
        """
        决定协同模式

        Args:
            unit: 当前单位
            target: 咬尾目标
            own_units: 我方所有单位
            enemy_units: 敌方所有单位
            tail_chase_states: 各单位咬尾状态

        Returns:
            (CooperationMode, 协同友机/None)
        """
        own_count = len(own_units)
        enemy_count = len(enemy_units)
        tail_chase_states = tail_chase_states or {}

        # 有人机特殊处理：不参与咬尾协同决策，让其回中心
        if unit.get('type') == '有人机':
            return CooperationMode.NO_ACTION, None

        # 查找附近友机
        nearby_teammate = self._find_nearby_teammate(unit, own_units)

        # 规则1: 我方≤3架时，禁止任何单机作战
        if own_count <= 3:
            if nearby_teammate:
                return CooperationMode.DUAL_ATTACK, nearby_teammate
            else:
                return CooperationMode.WAIT_SUPPORT, None

        # 规则2: 友机在附近 → 双机协同
        if nearby_teammate:
            return CooperationMode.DUAL_ATTACK, nearby_teammate

        # 规则3: 友机不在附近，检查是否应该单挑
        # 修改：不再使用RETREAT，改为WAIT_SUPPORT或允许单挑
        if enemy_count == 1:
            # 敌机1架 → 检查是否已有单挑
            if self._is_solo_slot_available(tail_chase_states, unit.get('name', '')):
                return CooperationMode.SOLO_ATTACK, None

        # 默认等待支援（在原地警戒而不是撤退）
        return CooperationMode.WAIT_SUPPORT, None

    def _find_nearby_teammate(self, unit: Dict, own_units: List[Dict]) -> Optional[Dict]:
        """查找附近的友机"""
        u_lon = unit.get('longitude', 0)
        u_lat = unit.get('latitude', 0)
        unit_name = unit.get('name', '')

        for teammate in own_units:
            if teammate.get('name', '') == unit_name:
                continue
            t_lon = teammate.get('longitude', 0)
            t_lat = teammate.get('latitude', 0)
            if t_lon and t_lat:
                dist = YxGeoUtils.haversine_distance(u_lon, u_lat, t_lon, t_lat)
                if dist < self.params.TEAMMATE_NEARBY_DISTANCE:
                    return teammate
        return None

    def _distance_to_main_force(self, unit: Dict, own_units: List[Dict]) -> float:
        """计算到大部队的距离（以重心为准）"""
        if len(own_units) <= 1:
            return 0

        u_lon = unit.get('longitude', 0)
        u_lat = unit.get('latitude', 0)
        unit_name = unit.get('name', '')

        # 计算除自己外的平均位置
        other_units = [u for u in own_units if u.get('name', '') != unit_name]
        if not other_units:
            return 0

        center_lon = sum(u.get('longitude', 0) for u in other_units) / len(other_units)
        center_lat = sum(u.get('latitude', 0) for u in other_units) / len(other_units)

        return YxGeoUtils.haversine_distance(u_lon, u_lat, center_lon, center_lat)

    def _is_solo_slot_available(self, tail_chase_states: Dict, unit_name: str) -> bool:
        """检查是否还有单挑名额（同一时间只允许1架）"""
        for name, state in tail_chase_states.items():
            if name == unit_name:
                continue
            if state.get('cooperation_mode') == CooperationMode.SOLO_ATTACK:
                return False  # 已有单挑，不允许更多
        return True


class TailChaseTactics:
    """
    咬尾战术计算器

    提供姿态角计算、绕飞目标点计算、动态半径调整等核心功能
    """

    def __init__(self, params: TailChaseParams = None):
        self.params = params or TailChaseParams()
        self.debug = False

    def calculate_aspect_angle(
        self,
        my_pos: Tuple[float, float],
        enemy_pos: Tuple[float, float],
        enemy_heading: float
    ) -> float:
        """
        计算我方相对敌机的姿态角

        Args:
            my_pos: 我方位置 (lon, lat)
            enemy_pos: 敌机位置 (lon, lat)
            enemy_heading: 敌机航向（弧度）

        Returns:
            姿态角 0-180度，180=完美尾追
        """
        # 敌机到我方的方位角
        bearing = YxGeoUtils.calculate_bearing(
            enemy_pos[0], enemy_pos[1],
            my_pos[0], my_pos[1]
        )

        # 敌机航向（转换为度）
        enemy_heading_deg = math.degrees(enemy_heading) % 360

        # 与敌机航向的夹角
        angle_diff = abs(bearing - enemy_heading_deg)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        return angle_diff

    def calculate_chase_point(
        self,
        enemy_pos: Tuple[float, float],
        enemy_heading: float,
        radius: float,
        phase_angle: float = 0
    ) -> Tuple[float, float]:
        """
        计算绕飞目标点

        Args:
            enemy_pos: 敌机位置 (lon, lat)
            enemy_heading: 敌机航向（弧度）
            radius: 绕飞半径（米）
            phase_angle: 相位角（度），0=敌机正后方，正值=顺时针偏移

        Returns:
            目标点坐标 (lon, lat)
        """
        # 敌机航向（度）
        enemy_heading_deg = math.degrees(enemy_heading) % 360

        # 敌机后方方向
        tail_direction = (enemy_heading_deg + 180) % 360

        # 加上相位角偏移
        target_direction = (tail_direction + phase_angle) % 360

        # 计算目标点
        radius_km = radius / 1000
        lon_off, lat_off = YxGeoUtils.km_to_lon_lat(
            enemy_pos[1], radius_km, target_direction
        )

        return (enemy_pos[0] + lon_off, enemy_pos[1] + lat_off)

    def adjust_radius(
        self,
        current_radius: float,
        aspect_angle: float,
        distance: float,
        target_is_manned: bool
    ) -> float:
        """
        根据态势动态调整绕飞半径

        Args:
            current_radius: 当前半径
            aspect_angle: 姿态角
            distance: 与敌机距离
            target_is_manned: 目标是否为有人机

        Returns:
            调整后的半径
        """
        target_radius = current_radius
        target_nez = self.params.MANNED_TAIL_NEZ if target_is_manned else self.params.UAV_TAIL_NEZ

        # 根据姿态角调整
        if aspect_angle < self.params.CHASE_ANGLE:
            # 姿态角不够，保持大半径继续调整
            target_radius = max(current_radius, self.params.RADIUS_CHASE)
        elif aspect_angle >= 150:
            # 姿态角很好，可以缩小半径
            target_radius = min(current_radius, self.params.RADIUS_ATTACK)

        # 根据距离微调
        if distance > current_radius * 1.3:
            # 距离太远，缩小半径加速接近
            target_radius *= self.params.RADIUS_SHRINK_RATE
        elif distance < current_radius * 0.8:
            # 距离太近，扩大半径
            target_radius *= self.params.RADIUS_EXPAND_RATE

        # 接近NEZ时进一步缩小
        if distance < target_nez * 1.5:
            target_radius = min(target_radius, self.params.RADIUS_MIN)

        # 限制范围
        return max(self.params.RADIUS_MIN, min(self.params.RADIUS_APPROACH, target_radius))

    def select_chase_direction(
        self,
        my_pos: Tuple[float, float],
        my_heading: float,
        enemy_pos: Tuple[float, float],
        enemy_heading: float
    ) -> str:
        """
        选择顺时针或逆时针绕飞

        原则：选择转弯角度小的方向

        Returns:
            "CW" = 顺时针, "CCW" = 逆时针
        """
        # 敌机尾部方向
        enemy_heading_deg = math.degrees(enemy_heading) % 360
        tail_dir = (enemy_heading_deg + 180) % 360

        # 我方到敌机的方位角
        current_bearing = YxGeoUtils.calculate_bearing(
            my_pos[0], my_pos[1],
            enemy_pos[0], enemy_pos[1]
        )

        # 计算顺时针和逆时针到达尾部的角度差
        cw_angle = (tail_dir - current_bearing + 360) % 360
        ccw_angle = (current_bearing - tail_dir + 360) % 360

        return "CW" if cw_angle < ccw_angle else "CCW"

    def get_target_speed(self, is_manned: bool, state: TailChaseState) -> float:
        """获取目标速度"""
        if is_manned:
            max_speed = self.params.MANNED_MAX_SPEED
            corner_speed = self.params.CORNER_SPEED_MANNED
        else:
            max_speed = self.params.UAV_MAX_SPEED
            corner_speed = self.params.CORNER_SPEED_UAV

        if state == TailChaseState.ATTACK:
            return corner_speed * self.params.ATTACK_SPEED_RATIO
        elif state == TailChaseState.CHASE:
            return corner_speed
        else:
            return max_speed * self.params.CHASE_SPEED_RATIO

    def should_fire(
        self,
        aspect_angle: float,
        distance: float,
        target_is_manned: bool,
        cooperation_mode: CooperationMode
    ) -> Tuple[bool, float]:
        """
        判断是否应该开火

        Args:
            aspect_angle: 姿态角
            distance: 距离
            target_is_manned: 目标是否为有人机
            cooperation_mode: 协同模式

        Returns:
            (是否开火, 估算Pk)
        """
        # 姿态角条件
        if aspect_angle < self.params.CHASE_ANGLE:
            return False, 0.0

        # 计算尾追NEZ
        tail_nez = self.params.MANNED_TAIL_NEZ if target_is_manned else self.params.UAV_TAIL_NEZ

        # 距离条件
        if distance > tail_nez:
            return False, 0.0

        # 根据协同模式选择阈值
        if cooperation_mode == CooperationMode.DUAL_ATTACK:
            pk_threshold = self.params.PK_THRESHOLD_DUAL
        elif cooperation_mode == CooperationMode.SOLO_ATTACK:
            pk_threshold = self.params.PK_THRESHOLD_SOLO
        else:
            pk_threshold = self.params.PK_THRESHOLD_DEFAULT

        # 估算Pk
        pk = self._estimate_pk(aspect_angle, distance, tail_nez)

        return pk >= pk_threshold, pk

    def _estimate_pk(self, aspect_angle: float, distance: float, nez: float) -> float:
        """
        估算命中概率

        基于姿态角和距离的简化模型
        """
        # 姿态角因子（120-180度映射到0.5-1.0）
        angle_factor = min(1.0, max(0.5, (aspect_angle - 120) / 60 * 0.5 + 0.5))

        # 距离因子（NEZ内越近越好）
        dist_factor = max(0.3, 1.0 - distance / nez * 0.7)

        return angle_factor * dist_factor


class ActionTailChase(Action):
    """
    咬尾机动行为树节点

    实现完整的咬尾状态机：
    SEARCH → APPROACH → CHASE → ATTACK → DISENGAGE
    """

    DEBUG_ENABLED = False
    DEBUG_INTERVAL = 30

    def __init__(self, params: TailChaseParams = None):
        self.params = params or TailChaseParams()
        self.tactics = TailChaseTactics(self.params)
        self.coordinator = TailChaseCoordinator(self.params)

        # 每个单位的咬尾状态
        self._unit_states: Dict[str, Dict] = {}

    def tick(self, agent) -> str:
        """执行咬尾机动"""
        # 没有敌机时跳过
        if not agent.enemy_units:
            return NodeStatus.SUCCESS

        # 获取可用单位
        available_units = self._get_available_units(agent)
        if not available_units:
            return NodeStatus.SUCCESS

        # 对每个可用单位执行咬尾逻辑
        for unit in available_units:
            self._execute_tail_chase(agent, unit)

        return NodeStatus.SUCCESS

    def _get_available_units(self, agent) -> List[Dict]:
        """获取未被其他节点控制的单位"""
        available = []
        for unit in agent.own_units:
            unit_name = unit.get('name', '')
            # 跳过已被控制的单位
            if unit_name not in agent.commanded_units:
                if unit.get('longitude') and unit.get('latitude'):
                    available.append(unit)
        return available

    def _execute_tail_chase(self, agent, unit: Dict):
        """对单个单位执行咬尾逻辑"""
        unit_name = unit.get('name', '')
        is_manned = unit.get('type') == '有人机'

        # 有人机特殊处理：只在安全情况下参与咬尾
        if is_manned:
            if not self._is_safe_for_manned_attack(agent, unit):
                # 有人机不安全，跳过咬尾，让后续节点（中心优先）处理
                return

        # 获取或初始化状态
        if unit_name not in self._unit_states:
            self._unit_states[unit_name] = {
                'state': TailChaseState.SEARCH,
                'target_enemy': None,
                'cooperation_mode': CooperationMode.NO_ACTION,
                'teammate': None,
                'current_radius': self.params.RADIUS_APPROACH,
                'chase_direction': 'CW',
                'phase_angle': 0,
                'state_start_frame': agent.frame_count,
                'last_aspect_angle': 0,
                'last_distance': float('inf')
            }

        state = self._unit_states[unit_name]

        # 状态机处理
        if state['state'] == TailChaseState.SEARCH:
            self._handle_search(agent, unit, state)
        elif state['state'] == TailChaseState.APPROACH:
            self._handle_approach(agent, unit, state)
        elif state['state'] == TailChaseState.CHASE:
            self._handle_chase(agent, unit, state)
        elif state['state'] == TailChaseState.ATTACK:
            self._handle_attack(agent, unit, state)
        elif state['state'] == TailChaseState.DISENGAGE:
            self._handle_disengage(agent, unit, state)

    def _handle_search(self, agent, unit: Dict, state: Dict):
        """搜索阶段：寻找适合咬尾的目标"""
        unit_name = unit.get('name', '')
        u_lon = unit.get('longitude', 0)
        u_lat = unit.get('latitude', 0)

        # 寻找最近的敌机
        best_target = None
        min_dist = float('inf')

        for enemy in agent.enemy_units:
            e_lon = enemy.get('longitude', 0)
            e_lat = enemy.get('latitude', 0)
            if e_lon and e_lat:
                dist = YxGeoUtils.haversine_distance(u_lon, u_lat, e_lon, e_lat)
                if dist < self.params.SEARCH_RANGE and dist < min_dist:
                    min_dist = dist
                    best_target = enemy

        if best_target:
            # 决定协同模式
            coop_mode, teammate = self.coordinator.decide_cooperation_mode(
                unit, best_target,
                agent.own_units, agent.enemy_units,
                self._unit_states
            )

            # 根据协同模式决定行动
            if coop_mode == CooperationMode.RETREAT:
                # 撤退：飞向大部队
                self._execute_retreat(agent, unit)
                return
            elif coop_mode == CooperationMode.WAIT_SUPPORT:
                # 等待支援：保持距离，不进攻
                self._execute_wait_support(agent, unit, best_target)
                return
            elif coop_mode in [CooperationMode.DUAL_ATTACK, CooperationMode.SOLO_ATTACK]:
                # 可以进攻：转换到接近阶段
                state['target_enemy'] = best_target.get('name', '')
                state['cooperation_mode'] = coop_mode
                state['teammate'] = teammate
                state['state'] = TailChaseState.APPROACH
                state['state_start_frame'] = agent.frame_count

                # 选择绕飞方向
                state['chase_direction'] = self.tactics.select_chase_direction(
                    (u_lon, u_lat),
                    unit.get('heading', 0),
                    (e_lon, e_lat),
                    best_target.get('heading', 0)
                )

                if self.DEBUG_ENABLED:
                    print(f"[咬尾] {unit_name} 进入APPROACH, 目标: {state['target_enemy']}, "
                          f"模式: {coop_mode.value}")

    def _handle_approach(self, agent, unit: Dict, state: Dict):
        """接近阶段：调整角度进入敌机后半球"""
        unit_name = unit.get('name', '')
        u_lon = unit.get('longitude', 0)
        u_lat = unit.get('latitude', 0)

        # 查找目标敌机
        target = self._find_target_enemy(agent, state['target_enemy'])
        if not target:
            state['state'] = TailChaseState.SEARCH
            return

        e_lon = target.get('longitude', 0)
        e_lat = target.get('latitude', 0)
        e_heading = target.get('heading', 0)

        # 计算姿态角和距离
        aspect_angle = self.tactics.calculate_aspect_angle(
            (u_lon, u_lat), (e_lon, e_lat), e_heading
        )
        distance = YxGeoUtils.haversine_distance(u_lon, u_lat, e_lon, e_lat)

        state['last_aspect_angle'] = aspect_angle
        state['last_distance'] = distance

        # 检查超时
        if agent.frame_count - state['state_start_frame'] > self.params.APPROACH_TIMEOUT:
            state['state'] = TailChaseState.DISENGAGE
            state['state_start_frame'] = agent.frame_count
            return

        # 检查转换条件
        if aspect_angle > self.params.APPROACH_ANGLE and distance < self.params.APPROACH_RANGE:
            state['state'] = TailChaseState.CHASE
            state['state_start_frame'] = agent.frame_count
            state['current_radius'] = self.params.RADIUS_CHASE
            if self.DEBUG_ENABLED:
                print(f"[咬尾] {unit_name} 进入CHASE, 姿态角: {aspect_angle:.1f}°")
            return

        # 执行接近机动
        self._execute_approach_maneuver(agent, unit, target, state)

    def _handle_chase(self, agent, unit: Dict, state: Dict):
        """追踪阶段：绕飞咬尾，逐步缩小半径"""
        unit_name = unit.get('name', '')
        u_lon = unit.get('longitude', 0)
        u_lat = unit.get('latitude', 0)

        # 查找目标敌机
        target = self._find_target_enemy(agent, state['target_enemy'])
        if not target:
            state['state'] = TailChaseState.SEARCH
            return

        e_lon = target.get('longitude', 0)
        e_lat = target.get('latitude', 0)
        e_heading = target.get('heading', 0)
        target_is_manned = target.get('type') == '有人机'

        # 计算姿态角和距离
        aspect_angle = self.tactics.calculate_aspect_angle(
            (u_lon, u_lat), (e_lon, e_lat), e_heading
        )
        distance = YxGeoUtils.haversine_distance(u_lon, u_lat, e_lon, e_lat)

        state['last_aspect_angle'] = aspect_angle
        state['last_distance'] = distance

        # 检查超时
        if agent.frame_count - state['state_start_frame'] > self.params.CHASE_TIMEOUT:
            state['state'] = TailChaseState.DISENGAGE
            state['state_start_frame'] = agent.frame_count
            return

        # 检查目标是否脱离
        if distance > self.params.SEARCH_RANGE:
            state['state'] = TailChaseState.DISENGAGE
            state['state_start_frame'] = agent.frame_count
            return

        # 动态调整半径
        state['current_radius'] = self.tactics.adjust_radius(
            state['current_radius'], aspect_angle, distance, target_is_manned
        )

        # 检查攻击条件
        tail_nez = self.params.MANNED_TAIL_NEZ if target_is_manned else self.params.UAV_TAIL_NEZ
        if aspect_angle > self.params.CHASE_ANGLE and distance < tail_nez:
            state['state'] = TailChaseState.ATTACK
            state['state_start_frame'] = agent.frame_count
            if self.DEBUG_ENABLED:
                print(f"[咬尾] {unit_name} 进入ATTACK, 姿态角: {aspect_angle:.1f}°, "
                      f"距离: {distance/1000:.1f}km")
            return

        # 执行追踪机动
        self._execute_chase_maneuver(agent, unit, target, state)

    def _handle_attack(self, agent, unit: Dict, state: Dict):
        """攻击阶段：稳定跟踪并开火"""
        unit_name = unit.get('name', '')
        u_lon = unit.get('longitude', 0)
        u_lat = unit.get('latitude', 0)

        # 查找目标敌机
        target = self._find_target_enemy(agent, state['target_enemy'])
        if not target:
            state['state'] = TailChaseState.SEARCH
            return

        e_lon = target.get('longitude', 0)
        e_lat = target.get('latitude', 0)
        e_heading = target.get('heading', 0)
        target_is_manned = target.get('type') == '有人机'

        # 计算姿态角和距离
        aspect_angle = self.tactics.calculate_aspect_angle(
            (u_lon, u_lat), (e_lon, e_lat), e_heading
        )
        distance = YxGeoUtils.haversine_distance(u_lon, u_lat, e_lon, e_lat)

        state['last_aspect_angle'] = aspect_angle
        state['last_distance'] = distance

        # 检查是否脱离攻击位置
        tail_nez = self.params.MANNED_TAIL_NEZ if target_is_manned else self.params.UAV_TAIL_NEZ
        if aspect_angle < self.params.APPROACH_ANGLE or distance > self.params.SEARCH_RANGE:
            state['state'] = TailChaseState.DISENGAGE
            state['state_start_frame'] = agent.frame_count
            return

        # 如果脱离NEZ但仍在后半球，回到CHASE
        if distance > tail_nez * 1.2 and aspect_angle > self.params.APPROACH_ANGLE:
            state['state'] = TailChaseState.CHASE
            state['state_start_frame'] = agent.frame_count
            return

        # 判断是否开火（通过 fire_control 模块处理）
        should_fire, pk = self.tactics.should_fire(
            aspect_angle, distance, target_is_manned, state['cooperation_mode']
        )

        if should_fire:
            # 设置攻击优先级，让 fire_control 模块处理实际开火
            if hasattr(agent, 'tail_chase_fire_targets'):
                agent.tail_chase_fire_targets[unit_name] = {
                    'target': target,
                    'pk': pk,
                    'aspect_angle': aspect_angle,
                    'distance': distance,
                    'cooperation_mode': state['cooperation_mode']
                }

            if self.DEBUG_ENABLED and agent.frame_count % self.DEBUG_INTERVAL == 0:
                print(f"[咬尾] {unit_name} 建议开火, Pk: {pk:.2f}, "
                      f"姿态角: {aspect_angle:.1f}°, 距离: {distance/1000:.1f}km")

        # 继续保持攻击位置
        self._execute_attack_maneuver(agent, unit, target, state)

    def _handle_disengage(self, agent, unit: Dict, state: Dict):
        """脱离阶段：重新寻找目标"""
        # 检查脱离持续时间
        if agent.frame_count - state['state_start_frame'] > self.params.DISENGAGE_DURATION:
            state['state'] = TailChaseState.SEARCH
            state['target_enemy'] = None
            state['cooperation_mode'] = CooperationMode.NO_ACTION
            state['teammate'] = None
            return

        # 执行脱离机动（远离当前区域）
        self._execute_disengage_maneuver(agent, unit, state)

    def _find_target_enemy(self, agent, target_name: str) -> Optional[Dict]:
        """根据名称查找目标敌机"""
        for enemy in agent.enemy_units:
            if enemy.get('name', '') == target_name:
                return enemy
        return None

    def _execute_approach_maneuver(self, agent, unit: Dict, target: Dict, state: Dict):
        """执行接近机动"""
        unit_name = unit.get('name', '')
        u_lat = unit.get('latitude', 0)
        is_manned = unit.get('type') == '有人机'

        e_lon = target.get('longitude', 0)
        e_lat = target.get('latitude', 0)
        e_alt = target.get('altitude', 4000)
        e_heading = target.get('heading', 0)

        # 计算目标点（敌机后方，大半径）
        phase_offset = 30 if state['chase_direction'] == 'CW' else -30
        target_pos = self.tactics.calculate_chase_point(
            (e_lon, e_lat), e_heading,
            self.params.RADIUS_APPROACH,
            phase_offset
        )

        # 目标高度和速度
        target_alt = min(self.params.MAX_ALTITUDE,
                        max(self.params.MIN_ALTITUDE, e_alt + self.params.ALTITUDE_ADVANTAGE))
        target_speed = self.tactics.get_target_speed(is_manned, TailChaseState.APPROACH)

        # 发送指令
        agent.add_action(
            decCmd.fly_to_point(
                unit_name,
                (target_pos[1], target_pos[0], target_alt),  # (lat, lon, alt)
                target_speed
            ),
            unit_name
        )

    def _execute_chase_maneuver(self, agent, unit: Dict, target: Dict, state: Dict):
        """执行追踪机动"""
        unit_name = unit.get('name', '')
        is_manned = unit.get('type') == '有人机'

        e_lon = target.get('longitude', 0)
        e_lat = target.get('latitude', 0)
        e_alt = target.get('altitude', 4000)
        e_heading = target.get('heading', 0)

        # 更新相位角（逐步向尾部移动）
        step = 10 if state['chase_direction'] == 'CW' else -10
        state['phase_angle'] = (state['phase_angle'] + step) % 360

        # 限制相位角范围（保持在后半球）
        if state['chase_direction'] == 'CW':
            if state['phase_angle'] > 60:
                state['phase_angle'] = -60
        else:
            if state['phase_angle'] < -60:
                state['phase_angle'] = 60

        # 计算目标点
        target_pos = self.tactics.calculate_chase_point(
            (e_lon, e_lat), e_heading,
            state['current_radius'],
            state['phase_angle']
        )

        # 目标高度和速度
        target_alt = min(self.params.MAX_ALTITUDE,
                        max(self.params.MIN_ALTITUDE, e_alt + self.params.ALTITUDE_ADVANTAGE))
        target_speed = self.tactics.get_target_speed(is_manned, TailChaseState.CHASE)

        # 调试输出
        if self.DEBUG_ENABLED and agent.frame_count % self.DEBUG_INTERVAL == 0:
            print(f"[咬尾] {unit_name} CHASE - "
                  f"半径: {state['current_radius']/1000:.1f}km, "
                  f"相位: {state['phase_angle']:.0f}°, "
                  f"姿态角: {state['last_aspect_angle']:.1f}°")

        # 发送指令
        agent.add_action(
            decCmd.fly_to_point(
                unit_name,
                (target_pos[1], target_pos[0], target_alt),
                target_speed
            ),
            unit_name
        )

    def _execute_attack_maneuver(self, agent, unit: Dict, target: Dict, state: Dict):
        """执行攻击机动"""
        unit_name = unit.get('name', '')
        is_manned = unit.get('type') == '有人机'

        e_lon = target.get('longitude', 0)
        e_lat = target.get('latitude', 0)
        e_alt = target.get('altitude', 4000)
        e_heading = target.get('heading', 0)

        # 攻击阶段保持小半径和稳定相位
        target_pos = self.tactics.calculate_chase_point(
            (e_lon, e_lat), e_heading,
            self.params.RADIUS_MIN,
            0  # 正后方
        )

        # 目标高度和速度
        target_alt = min(self.params.MAX_ALTITUDE,
                        max(self.params.MIN_ALTITUDE, e_alt + self.params.ALTITUDE_ADVANTAGE))
        target_speed = self.tactics.get_target_speed(is_manned, TailChaseState.ATTACK)

        # 发送指令
        agent.add_action(
            decCmd.fly_to_point(
                unit_name,
                (target_pos[1], target_pos[0], target_alt),
                target_speed
            ),
            unit_name
        )

    def _execute_disengage_maneuver(self, agent, unit: Dict, state: Dict):
        """执行脱离机动"""
        unit_name = unit.get('name', '')
        u_lon = unit.get('longitude', 0)
        u_lat = unit.get('latitude', 0)
        u_alt = unit.get('altitude', 4000)
        u_heading = unit.get('heading', 0)
        is_manned = unit.get('type') == '有人机'

        # 沿当前航向继续飞行
        heading_deg = math.degrees(u_heading) % 360
        disengage_dist_km = 5  # 脱离5km

        lon_off, lat_off = YxGeoUtils.km_to_lon_lat(u_lat, disengage_dist_km, heading_deg)
        target_lon = u_lon + lon_off
        target_lat = u_lat + lat_off
        target_alt = min(self.params.MAX_ALTITUDE, max(self.params.MIN_ALTITUDE, u_alt))

        max_speed = self.params.MANNED_MAX_SPEED if is_manned else self.params.UAV_MAX_SPEED

        agent.add_action(
            decCmd.fly_to_point(
                unit_name,
                (target_lat, target_lon, target_alt),
                max_speed * 0.9
            ),
            unit_name
        )

    def _execute_retreat(self, agent, unit: Dict):
        """执行撤退（飞向大部队）"""
        unit_name = unit.get('name', '')
        u_lon = unit.get('longitude', 0)
        u_lat = unit.get('latitude', 0)
        u_alt = unit.get('altitude', 4000)
        is_manned = unit.get('type') == '有人机'

        # 计算大部队位置
        other_units = [u for u in agent.own_units if u.get('name', '') != unit_name]
        if not other_units:
            return

        center_lon = sum(u.get('longitude', 0) for u in other_units) / len(other_units)
        center_lat = sum(u.get('latitude', 0) for u in other_units) / len(other_units)

        target_alt = min(self.params.MAX_ALTITUDE, max(self.params.MIN_ALTITUDE, u_alt))
        max_speed = self.params.MANNED_MAX_SPEED if is_manned else self.params.UAV_MAX_SPEED

        agent.add_action(
            decCmd.fly_to_point(
                unit_name,
                (center_lat, center_lon, target_alt),
                max_speed
            ),
            unit_name
        )

    def _execute_wait_support(self, agent, unit: Dict, target: Dict):
        """执行等待支援（保持距离）"""
        unit_name = unit.get('name', '')
        u_lon = unit.get('longitude', 0)
        u_lat = unit.get('latitude', 0)
        u_alt = unit.get('altitude', 4000)
        is_manned = unit.get('type') == '有人机'

        e_lon = target.get('longitude', 0)
        e_lat = target.get('latitude', 0)

        # 计算与敌机的距离
        distance = YxGeoUtils.haversine_distance(u_lon, u_lat, e_lon, e_lat)

        # 保持在搜索范围边缘（不靠太近也不跑太远）
        safe_distance = self.params.SEARCH_RANGE * 0.9  # 22.5km

        if distance < safe_distance:
            # 太近，后退
            bearing = YxGeoUtils.calculate_bearing(e_lon, e_lat, u_lon, u_lat)
            retreat_dist_km = (safe_distance - distance) / 1000 + 3
            lon_off, lat_off = YxGeoUtils.km_to_lon_lat(u_lat, retreat_dist_km, bearing)
            target_lon = u_lon + lon_off
            target_lat = u_lat + lat_off
        else:
            # 保持位置，围绕敌机缓慢移动
            bearing = YxGeoUtils.calculate_bearing(e_lon, e_lat, u_lon, u_lat)
            orbit_bearing = (bearing + 30) % 360  # 缓慢绕行
            orbit_dist_km = safe_distance / 1000
            lon_off, lat_off = YxGeoUtils.km_to_lon_lat(e_lat, orbit_dist_km, orbit_bearing)
            target_lon = e_lon + lon_off
            target_lat = e_lat + lat_off

        target_alt = min(self.params.MAX_ALTITUDE, max(self.params.MIN_ALTITUDE, u_alt))
        corner_speed = self.params.CORNER_SPEED_MANNED if is_manned else self.params.CORNER_SPEED_UAV

        agent.add_action(
            decCmd.fly_to_point(
                unit_name,
                (target_lat, target_lon, target_alt),
                corner_speed
            ),
            unit_name
        )

    def _is_safe_for_manned_attack(self, agent, unit: Dict) -> bool:
        """
        判断有人机是否可以安全攻击

        条件：
        1. 30km内只有一架敌机
        2. 该敌机无弹药（基于观测推断）

        Returns:
            True = 安全可以攻击, False = 不安全应回中心
        """
        u_lon = unit.get('longitude', 0)
        u_lat = unit.get('latitude', 0)

        # 检查30km内的敌机
        nearby_enemies = []
        for enemy in agent.enemy_units:
            e_lon = enemy.get('longitude', 0)
            e_lat = enemy.get('latitude', 0)
            if e_lon and e_lat:
                dist = YxGeoUtils.haversine_distance(u_lon, u_lat, e_lon, e_lat)
                if dist < 30000:  # 30km内
                    nearby_enemies.append(enemy)

        if not nearby_enemies:
            return False  # 没有近距离敌机，不需要咬尾

        if len(nearby_enemies) > 1:
            return False  # 多架敌机，有人机不参与

        # 只有一架敌机，检查是否有弹药
        enemy = nearby_enemies[0]
        has_ammo = self._enemy_has_ammo(enemy)

        return not has_ammo  # 无弹药才安全

    def _enemy_has_ammo(self, enemy: Dict) -> bool:
        """
        检查敌机是否有弹药（基于观测推断）

        注意：敌机的武器信息通常不可见，这里使用保守假设
        """
        # 方法1：检查敌机weapons字段（如果有）
        weapons = enemy.get('weapons', [])
        for w in weapons:
            if w.get('quantity', 0) > 0:
                return True

        # 如果没有weapons信息，保守假设有弹药
        if not weapons:
            return True

        return False


# ==================== 导出 ====================
__all__ = [
    'TailChaseState',
    'CooperationMode',
    'TailChaseParams',
    'TailChaseCoordinator',
    'TailChaseTactics',
    'ActionTailChase'
]
