"""
单环机动模块 (One-Circle Maneuver)

核心概念：
=========
单环机动是一种围绕敌机飞行的战术移动模式，而不是替代攻击或躲避。

工作原理：
=========
1. 绕着敌机飞一个圆圈
2. 当面向敌机时 → 进入攻击窗口，现有攻击逻辑判断是否开火
3. 当侧向敌机时 → 垂直于敌机的速度分量最大，配合现有躲避机制
4. 当背向敌机时 → 拉开距离，可以脱离或重新进入

优势：
=====
- 攻击后可以马上拉开距离
- 在距离敌方最近时，垂直速度分量最大，更容易躲避
- 形成"拉扯"的战术节奏

官方参数：
=========
- 有人机：速度180-500m/s，最大径向加速度15m/s²
- 无人机：速度120-360m/s，最大径向加速度15m/s²
"""

import math
from typing import Dict, List, Tuple, Optional
from ..bt_framework import Action, NodeStatus
from utilities.yxScriptTreeFunc import YxScriptTreeFunc as decCmd
from utilities.yxGeoUtils import YxGeoUtils


class CircleManeuverTactics:
    """
    单环机动战术计算器

    提供绕敌飞行的目标点计算
    """

    # ========== 官方飞机参数 ==========
    MANNED_MIN_SPEED = 180          # 有人机最小速度 m/s
    MANNED_MAX_SPEED = 500          # 有人机最大速度 m/s
    MANNED_MAX_RADIAL_ACC = 15      # 有人机最大径向加速度 m/s²

    UAV_MIN_SPEED = 120             # 无人机最小速度 m/s
    UAV_MAX_SPEED = 360             # 无人机最大速度 m/s
    UAV_MAX_RADIAL_ACC = 15         # 无人机最大径向加速度 m/s²

    # ========== 单环机动参数 ==========
    # 绕飞半径（距离敌机的距离）
    CIRCLE_RADIUS_MIN = 8000        # 最小绕飞半径 8km
    CIRCLE_RADIUS_MAX = 15000       # 最大绕飞半径 15km
    CIRCLE_RADIUS_DEFAULT = 10000   # 默认绕飞半径 10km

    # 每帧推进角度（控制绕飞速度）
    CIRCLE_STEP_DEGREES = 15        # 每帧推进15度

    # 高度保持
    MIN_ALTITUDE = 2000
    MAX_ALTITUDE = 7000
    ALTITUDE_OFFSET = 300           # 比敌机高300m

    # 最佳转弯速度
    CORNER_SPEED_MANNED = 350       # 有人机最佳转弯速度 m/s
    CORNER_SPEED_UAV = 250          # 无人机最佳转弯速度 m/s

    def __init__(self):
        self.debug = False

    def calculate_turn_radius(self, speed: float, radial_acc: float) -> float:
        """
        计算转弯半径

        公式: R = V² / a
        """
        if radial_acc <= 0:
            return float('inf')
        return (speed ** 2) / radial_acc

    def get_corner_speed(self, is_manned: bool) -> float:
        """获取最佳转弯速度"""
        return self.CORNER_SPEED_MANNED if is_manned else self.CORNER_SPEED_UAV

    def get_max_speed(self, is_manned: bool) -> float:
        """获取最大速度"""
        return self.MANNED_MAX_SPEED if is_manned else self.UAV_MAX_SPEED

    def calculate_circle_position(
        self,
        own_unit: Dict,
        enemy: Dict,
        current_angle: float,
        circle_radius: float = None,
        clockwise: bool = True
    ) -> Dict:
        """
        计算单环机动的下一个目标位置

        Args:
            own_unit: 己方单位
            enemy: 敌方单位
            current_angle: 当前在圆上的角度（0=敌机正前方，90=敌机右侧，180=敌机后方）
            circle_radius: 绕飞半径（米），默认使用DEFAULT值
            clockwise: 是否顺时针绕飞

        Returns:
            {
                'target_lon': 目标经度,
                'target_lat': 目标纬度,
                'target_alt': 目标高度,
                'target_speed': 目标速度,
                'next_angle': 下一帧的角度,
                'phase': 当前阶段 ('APPROACH', 'ATTACK_WINDOW', 'DISENGAGE')
            }
        """
        is_manned = own_unit.get('type') == '有人机'

        # 敌机位置和航向
        e_lon = enemy.get('longitude', 0)
        e_lat = enemy.get('latitude', 0)
        e_alt = enemy.get('altitude', 4000)
        e_heading = math.degrees(enemy.get('heading', 0)) % 360

        # 绕飞半径
        if circle_radius is None:
            circle_radius = self.CIRCLE_RADIUS_DEFAULT

        # 计算下一个角度
        step = self.CIRCLE_STEP_DEGREES if clockwise else -self.CIRCLE_STEP_DEGREES
        next_angle = (current_angle + step) % 360

        # 目标位置相对于敌机的方位角
        # 0度 = 敌机正前方，90度 = 敌机右侧，180度 = 敌机后方
        # 转换为绝对方位角
        target_bearing = (e_heading + next_angle) % 360

        # 计算目标点
        radius_km = circle_radius / 1000
        lon_off, lat_off = YxGeoUtils.km_to_lon_lat(e_lat, radius_km, target_bearing)
        target_lon = e_lon + lon_off
        target_lat = e_lat + lat_off

        # 目标高度：保持略高于敌机
        target_alt = e_alt + self.ALTITUDE_OFFSET
        target_alt = max(self.MIN_ALTITUDE, min(self.MAX_ALTITUDE, target_alt))

        # 目标速度：使用最佳转弯速度
        target_speed = self.get_corner_speed(is_manned)

        # 判断当前阶段
        phase = self._determine_phase(next_angle)

        return {
            'target_lon': target_lon,
            'target_lat': target_lat,
            'target_alt': target_alt,
            'target_speed': target_speed,
            'next_angle': next_angle,
            'phase': phase,
            'description': f'单环机动 - {phase} (角度: {next_angle:.0f}°)'
        }

    def _determine_phase(self, angle: float) -> str:
        """
        判断当前在圆上的阶段

        Args:
            angle: 当前角度（0=敌机正前方）

        Returns:
            'APPROACH': 接近阶段（135-225度，在敌机后方）
            'ATTACK_WINDOW': 攻击窗口（315-45度，面向敌机）
            'DISENGAGE': 脱离阶段（45-135度 或 225-315度，侧向敌机）
        """
        # 归一化角度到0-360
        angle = angle % 360

        if 315 <= angle or angle <= 45:
            # 面向敌机 - 攻击窗口
            return 'ATTACK_WINDOW'
        elif 135 <= angle <= 225:
            # 在敌机后方 - 接近阶段（敌机看不到我们）
            return 'APPROACH'
        else:
            # 侧向敌机 - 脱离阶段（垂直速度最大）
            return 'DISENGAGE'

    def calculate_initial_angle(self, own_unit: Dict, enemy: Dict) -> float:
        """
        计算己方单位当前相对于敌机的角度

        用于初始化单环机动时确定起始位置
        """
        u_lon = own_unit.get('longitude', 0)
        u_lat = own_unit.get('latitude', 0)

        e_lon = enemy.get('longitude', 0)
        e_lat = enemy.get('latitude', 0)
        e_heading = math.degrees(enemy.get('heading', 0)) % 360

        # 计算我方相对于敌机的方位角
        bearing_from_enemy = YxGeoUtils.calculate_bearing(e_lon, e_lat, u_lon, u_lat)

        # 转换为相对于敌机航向的角度
        relative_angle = (bearing_from_enemy - e_heading + 360) % 360

        return relative_angle

    def should_use_circle_maneuver(
        self,
        own_unit: Dict,
        enemy: Dict,
        min_distance: float = 5000,
        max_distance: float = 25000
    ) -> bool:
        """
        判断是否应该使用单环机动

        条件：
        1. 与敌机距离在合适范围内（5-25km）
        2. 敌机不是正在逃跑

        Args:
            own_unit: 己方单位
            enemy: 敌方单位
            min_distance: 最小启用距离
            max_distance: 最大启用距离
        """
        u_lon = own_unit.get('longitude', 0)
        u_lat = own_unit.get('latitude', 0)

        e_lon = enemy.get('longitude', 0)
        e_lat = enemy.get('latitude', 0)

        # 计算距离
        distance = YxGeoUtils.haversine_distance(u_lon, u_lat, e_lon, e_lat)

        return min_distance <= distance <= max_distance

    def select_circle_direction(self, own_unit: Dict, enemy: Dict, agent) -> bool:
        """
        选择绕飞方向（顺时针或逆时针）

        策略：选择更靠近战场中心的方向

        Returns:
            True = 顺时针, False = 逆时针
        """
        u_lon = own_unit.get('longitude', 0)
        u_lat = own_unit.get('latitude', 0)

        e_lon = enemy.get('longitude', 0)
        e_lat = enemy.get('latitude', 0)
        e_heading = math.degrees(enemy.get('heading', 0)) % 360

        center_lon = getattr(agent, 'center_lon', e_lon)
        center_lat = getattr(agent, 'center_lat', e_lat)

        # 计算顺时针和逆时针方向的下一个位置
        radius_km = self.CIRCLE_RADIUS_DEFAULT / 1000

        # 顺时针：当前角度 + 90度
        current_angle = self.calculate_initial_angle(own_unit, enemy)
        cw_angle = (current_angle + 90) % 360
        ccw_angle = (current_angle - 90 + 360) % 360

        cw_bearing = (e_heading + cw_angle) % 360
        ccw_bearing = (e_heading + ccw_angle) % 360

        cw_lon_off, cw_lat_off = YxGeoUtils.km_to_lon_lat(e_lat, radius_km, cw_bearing)
        ccw_lon_off, ccw_lat_off = YxGeoUtils.km_to_lon_lat(e_lat, radius_km, ccw_bearing)

        cw_pos = (e_lon + cw_lon_off, e_lat + cw_lat_off)
        ccw_pos = (e_lon + ccw_lon_off, e_lat + ccw_lat_off)

        # 选择更靠近战场中心的方向
        cw_dist_to_center = YxGeoUtils.haversine_distance(cw_pos[0], cw_pos[1], center_lon, center_lat)
        ccw_dist_to_center = YxGeoUtils.haversine_distance(ccw_pos[0], ccw_pos[1], center_lon, center_lat)

        return cw_dist_to_center < ccw_dist_to_center


# ==================== 行为树动作节点 ====================

class ActionCircleManeuver(Action):
    """
    单环机动动作节点 - 基础版本

    只负责绕敌飞行，不干预攻击和躲避逻辑
    攻击和躲避由现有的节点处理
    """

    DEBUG_ENABLED = False
    DEBUG_INTERVAL = 30

    def __init__(self):
        self.tactics = CircleManeuverTactics()
        self.tactics.debug = self.DEBUG_ENABLED

        # 每个单位的绕飞状态
        self._unit_states = {}  # {unit_name: {'angle': float, 'clockwise': bool, 'target_enemy': str}}

    def tick(self, agent) -> str:
        """执行单环机动"""
        # 没有敌机时跳过
        if not agent.enemy_units:
            return NodeStatus.SUCCESS

        # 获取可用单位（未被其他节点控制的）
        available_units = self._get_available_units(agent)
        if not available_units:
            return NodeStatus.SUCCESS

        # 对每个可用单位执行单环机动
        for unit in available_units:
            self._execute_circle_for_unit(agent, unit)

        return NodeStatus.SUCCESS

    def _get_available_units(self, agent) -> List[Dict]:
        """获取未被其他节点控制的单位"""
        available = []
        for unit in agent.own_units:
            unit_name = unit.get('name', '')
            # 跳过已被控制的单位（如躲避导弹的单位）
            if unit_name not in agent.commanded_units:
                if unit.get('longitude') and unit.get('latitude'):
                    available.append(unit)
        return available

    def _select_target_enemy(self, agent, unit: Dict) -> Optional[Dict]:
        """为单位选择目标敌机（最近的敌机）"""
        if not agent.enemy_units:
            return None

        u_lon = unit.get('longitude', 0)
        u_lat = unit.get('latitude', 0)

        min_dist = float('inf')
        closest_enemy = None

        for enemy in agent.enemy_units:
            e_lon = enemy.get('longitude', 0)
            e_lat = enemy.get('latitude', 0)
            if e_lon and e_lat:
                dist = YxGeoUtils.haversine_distance(u_lon, u_lat, e_lon, e_lat)
                if dist < min_dist:
                    min_dist = dist
                    closest_enemy = enemy

        return closest_enemy

    def _execute_circle_for_unit(self, agent, unit: Dict):
        """对单个单位执行单环机动"""
        unit_name = unit.get('name', '')

        # 选择目标敌机
        enemy = self._select_target_enemy(agent, unit)
        if not enemy:
            return

        # 检查是否应该使用单环机动
        if not self.tactics.should_use_circle_maneuver(unit, enemy):
            # 距离不合适，清除状态
            if unit_name in self._unit_states:
                del self._unit_states[unit_name]
            return

        # 初始化或获取单位状态
        if unit_name not in self._unit_states:
            # 新单位，初始化状态
            initial_angle = self.tactics.calculate_initial_angle(unit, enemy)
            clockwise = self.tactics.select_circle_direction(unit, enemy, agent)
            self._unit_states[unit_name] = {
                'angle': initial_angle,
                'clockwise': clockwise,
                'target_enemy': enemy.get('name', '')
            }

        state = self._unit_states[unit_name]

        # 检查目标敌机是否变化
        if state['target_enemy'] != enemy.get('name', ''):
            # 目标变化，重新初始化
            state['angle'] = self.tactics.calculate_initial_angle(unit, enemy)
            state['clockwise'] = self.tactics.select_circle_direction(unit, enemy, agent)
            state['target_enemy'] = enemy.get('name', '')

        # 计算下一个位置
        maneuver = self.tactics.calculate_circle_position(
            unit, enemy,
            current_angle=state['angle'],
            clockwise=state['clockwise']
        )

        # 更新状态
        state['angle'] = maneuver['next_angle']

        # 调试输出
        if self.DEBUG_ENABLED and agent.frame_count % self.DEBUG_INTERVAL == 0:
            print(f"\n[单环机动] {unit_name}")
            print(f"  目标敌机: {enemy.get('name', 'unknown')}")
            print(f"  当前角度: {state['angle']:.0f}°")
            print(f"  阶段: {maneuver['phase']}")
            print(f"  方向: {'顺时针' if state['clockwise'] else '逆时针'}")

        # 发送移动指令
        agent.add_action(
            decCmd.fly_to_point(
                unit_name,
                (maneuver['target_lat'], maneuver['target_lon'], maneuver['target_alt']),
                maneuver['target_speed']
            ),
            unit_name
        )


class ActionAdaptiveManeuver(ActionCircleManeuver):
    """
    自适应单环机动

    在基础单环机动上增加：
    1. 根据敌机行为动态调整绕飞半径
    2. 根据态势切换绕飞方向
    3. 与攻击/躲避节点更好配合
    """

    # 动态调整参数
    RADIUS_ADJUST_STEP = 1000       # 半径调整步长 1km
    DIRECTION_SWITCH_COOLDOWN = 100  # 方向切换冷却（帧）

    def __init__(self):
        super().__init__()
        self._direction_switch_frame = {}  # {unit_name: last_switch_frame}
        self._current_radius = {}          # {unit_name: current_radius}

    def _execute_circle_for_unit(self, agent, unit: Dict):
        """对单个单位执行自适应单环机动"""
        unit_name = unit.get('name', '')

        # 选择目标敌机
        enemy = self._select_target_enemy(agent, unit)
        if not enemy:
            return

        # 检查是否应该使用单环机动
        if not self.tactics.should_use_circle_maneuver(unit, enemy):
            if unit_name in self._unit_states:
                del self._unit_states[unit_name]
            return

        # 初始化或获取单位状态
        if unit_name not in self._unit_states:
            initial_angle = self.tactics.calculate_initial_angle(unit, enemy)
            clockwise = self.tactics.select_circle_direction(unit, enemy, agent)
            self._unit_states[unit_name] = {
                'angle': initial_angle,
                'clockwise': clockwise,
                'target_enemy': enemy.get('name', '')
            }
            self._current_radius[unit_name] = self.tactics.CIRCLE_RADIUS_DEFAULT
            self._direction_switch_frame[unit_name] = 0

        state = self._unit_states[unit_name]

        # 动态调整半径
        radius = self._adjust_radius(agent, unit, enemy)

        # 检查是否需要切换方向
        self._check_direction_switch(agent, unit, enemy, state)

        # 计算下一个位置
        maneuver = self.tactics.calculate_circle_position(
            unit, enemy,
            current_angle=state['angle'],
            circle_radius=radius,
            clockwise=state['clockwise']
        )

        # 更新状态
        state['angle'] = maneuver['next_angle']

        # 调试输出
        if self.DEBUG_ENABLED and agent.frame_count % self.DEBUG_INTERVAL == 0:
            print(f"\n[自适应单环] {unit_name}")
            print(f"  目标敌机: {enemy.get('name', 'unknown')}")
            print(f"  角度: {state['angle']:.0f}°, 阶段: {maneuver['phase']}")
            print(f"  半径: {radius/1000:.1f}km, 方向: {'顺时针' if state['clockwise'] else '逆时针'}")

        # 发送移动指令
        agent.add_action(
            decCmd.fly_to_point(
                unit_name,
                (maneuver['target_lat'], maneuver['target_lon'], maneuver['target_alt']),
                maneuver['target_speed']
            ),
            unit_name
        )

    def _adjust_radius(self, agent, unit: Dict, enemy: Dict) -> float:
        """动态调整绕飞半径"""
        unit_name = unit.get('name', '')

        # 获取当前半径
        if unit_name not in self._current_radius:
            self._current_radius[unit_name] = self.tactics.CIRCLE_RADIUS_DEFAULT

        current_radius = self._current_radius[unit_name]

        # 检查有无导弹威胁
        has_missile_threat = self._check_missile_threat(agent, unit)

        if has_missile_threat:
            # 有导弹威胁，增大半径以拉开距离
            current_radius = min(
                self.tactics.CIRCLE_RADIUS_MAX,
                current_radius + self.RADIUS_ADJUST_STEP
            )
        else:
            # 无威胁，逐渐恢复默认半径
            if current_radius > self.tactics.CIRCLE_RADIUS_DEFAULT:
                current_radius -= self.RADIUS_ADJUST_STEP / 2
            elif current_radius < self.tactics.CIRCLE_RADIUS_DEFAULT:
                current_radius += self.RADIUS_ADJUST_STEP / 2

        self._current_radius[unit_name] = current_radius
        return current_radius

    def _check_missile_threat(self, agent, unit: Dict) -> bool:
        """检查是否有导弹威胁"""
        u_lon = unit.get('longitude', 0)
        u_lat = unit.get('latitude', 0)

        for missile in getattr(agent, 'enemy_missiles', []):
            m_lon = missile.get('longitude', 0)
            m_lat = missile.get('latitude', 0)

            if m_lon and m_lat:
                dist = YxGeoUtils.haversine_distance(u_lon, u_lat, m_lon, m_lat)
                if dist < 30000:  # 30km内有导弹
                    # 检查导弹是否指向我方
                    m_heading = math.degrees(missile.get('heading', 0)) % 360
                    bearing_to_me = YxGeoUtils.calculate_bearing(m_lon, m_lat, u_lon, u_lat)
                    angle_diff = abs((bearing_to_me - m_heading + 180) % 360 - 180)

                    if angle_diff < 45:  # 导弹指向我方
                        return True

        return False

    def _check_direction_switch(self, agent, unit: Dict, enemy: Dict, state: Dict):
        """检查是否需要切换绕飞方向"""
        unit_name = unit.get('name', '')

        # 检查冷却
        last_switch = self._direction_switch_frame.get(unit_name, 0)
        if agent.frame_count - last_switch < self.DIRECTION_SWITCH_COOLDOWN:
            return

        # 重新评估最佳方向
        best_direction = self.tactics.select_circle_direction(unit, enemy, agent)

        if best_direction != state['clockwise']:
            state['clockwise'] = best_direction
            self._direction_switch_frame[unit_name] = agent.frame_count

            if self.DEBUG_ENABLED:
                print(f"[自适应单环] {unit_name} 切换方向为: {'顺时针' if best_direction else '逆时针'}")
