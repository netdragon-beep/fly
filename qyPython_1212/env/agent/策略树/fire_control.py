"""
智能火控系统 (Smart Fire Control) - V2 物理增强版

核心概念：
1. WEZ (Weapon Engagement Zone) - 武器可攻击区，最大射程内
2. NEZ (No Escape Zone) - 不可逃逸区，目标无法规避的范围
3. Pk (Kill Probability) - 命中概率估算

V2增强：
- 基于真实物理参数的NEZ计算（导弹速度、飞机最大速度）
- 考虑目标相对每架射手的径向速度（接近/远离）
- 考虑目标相对每架射手的横向速度（逃逸速度）
- 双机协同时综合评估两个方向的逃逸难度

算法原理：
- NEZ = 导弹在目标逃逸前能追上的最大距离
- 目标逃逸速度越大，NEZ越小
- 双机从不同方向攻击时，目标无法同时向两个方向逃逸
"""

import math
from utilities.yxGeoUtils import YxGeoUtils


class SmartFireControl:
    """
    智能火控系统 V2 - 物理增强版

    === 官方参数（来自初赛赛题任务想定设计.pdf）===
    - 导弹：速度1200m/s，最大飞行时间60s，最大飞行距离72km
    - 有人机：速度180-500m/s，高度2000-7000m
    - 无人机：速度120-360m/s，高度2000-7000m
    """

    # === 飞机性能参数（官方参数）===
    # 有人机
    MANNED_MAX_SPEED = 500         # 有人机最大速度 m/s（官方：180-500）
    MANNED_CRUISE_SPEED = 400      # 有人机巡航速度 m/s

    # 无人机
    UAV_MAX_SPEED = 360            # 无人机最大速度 m/s（官方：120-360）
    UAV_CRUISE_SPEED = 300         # 无人机巡航速度 m/s

    # === 导弹参数（官方参数）===
    # 统一导弹参数（有人机和无人机使用相同导弹）
    MISSILE_SPEED = 1200           # 导弹平均速度 m/s（官方：1200）
    MISSILE_MAX_RANGE = 72000      # 导弹最大射程 m（官方：72km）
    MISSILE_MAX_DURATION = 60      # 导弹最大飞行时间 s（官方：60s）
    MISSILE_KILL_RADIUS = 10       # 导弹杀伤半径 m（官方：10m）

    # 有人机导弹（与无人机相同，保留分类便于未来扩展）
    MANNED_MISSILE_SPEED = 1200    # 导弹平均速度 m/s
    MANNED_MISSILE_RANGE = 72000   # 导弹最大射程 72km（官方参数）
    MANNED_MISSILE_DURATION = 60   # 导弹最大飞行时间 s

    # 无人机导弹
    UAV_MISSILE_SPEED = 1200       # 导弹平均速度 m/s
    UAV_MISSILE_RANGE = 72000      # 导弹最大射程 72km（官方参数）
    UAV_MISSILE_DURATION = 60      # 导弹最大飞行时间 s

    # === 武器射程参数（实战有效射程，小于理论最大射程）===
    # 考虑到目标会规避，实际有效射程约为理论射程的30-50%
    MANNED_MAX_RANGE = 35000       # 有人机有效射程 35km（考虑目标规避）
    MANNED_NEZ_HEAD_ON = 20000     # 迎头NEZ 20km（目标迎面飞来时）
    MANNED_NEZ_TAIL = 8000         # 尾追NEZ 8km（目标逃跑时）
    MANNED_OPTIMAL_RANGE = 25000   # 最优射程 25km

    UAV_MAX_RANGE = 30000          # 无人机有效射程 30km
    UAV_NEZ_HEAD_ON = 18000        # 迎头NEZ 18km
    UAV_NEZ_TAIL = 6000            # 尾追NEZ 6km
    UAV_OPTIMAL_RANGE = 20000      # 最优射程 20km

    # === 开火阈值 ===
    PK_THRESHOLD_NORMAL = 0.45     # 正常情况下的Pk阈值
    PK_THRESHOLD_URGENT = 0.40     # 紧急情况的Pk阈值
    PK_THRESHOLD_HIGH_VALUE = 0.40 # 高价值目标（有人机）的Pk阈值
    PK_THRESHOLD_COORDINATED = 0.35  # 双机协同开火阈值（更低，因为双发命中率高）

    # === 姿态角限制 ===
    MAX_ASPECT_FOR_NEZ_FIRE = 70   # NEZ内开火最大姿态角
    MAX_ASPECT_FOR_FLEEING = 90    # 目标逃跑时最大姿态角

    # === 调试开关 ===
    DEBUG_ENABLED = True
    DEBUG_INTERVAL = 30            # 每N帧输出一次

    @classmethod
    def calculate_aspect_angle(cls, shooter_lon, shooter_lat, shooter_heading,
                                target_lon, target_lat, target_heading):
        """
        计算姿态角 (Aspect Angle)

        姿态角定义：从目标视角看，射手相对于目标航向的角度
        - 0° = 迎头 (head-on)，目标正对着我们飞来
        - 180° = 尾追 (tail-chase)，目标背对我们逃跑
        - 90° = 横越 (beam)，目标横向通过

        返回: 0-180度
        """
        # 目标到射手的方位角
        bearing_to_shooter = YxGeoUtils.calculate_bearing(
            target_lon, target_lat, shooter_lon, shooter_lat
        )

        # 目标航向（弧度转角度）
        target_hdg = math.degrees(target_heading) % 360 if isinstance(target_heading, float) else target_heading % 360

        # 姿态角 = 目标航向与目标到射手方位的夹角
        aspect = abs((bearing_to_shooter - target_hdg + 180) % 360 - 180)

        return aspect

    @classmethod
    def calculate_closure_rate(cls, shooter_lon, shooter_lat, shooter_speed, shooter_heading,
                                target_lon, target_lat, target_speed, target_heading):
        """
        计算接近率 (Closure Rate)

        正值 = 双方接近
        负值 = 双方远离
        单位: m/s
        """
        # 射手到目标的方位
        bearing_to_target = YxGeoUtils.calculate_bearing(
            shooter_lon, shooter_lat, target_lon, target_lat
        )

        # 射手航向
        shooter_hdg = math.degrees(shooter_heading) % 360 if isinstance(shooter_heading, float) else shooter_heading % 360
        # 目标航向
        target_hdg = math.degrees(target_heading) % 360 if isinstance(target_heading, float) else target_heading % 360

        # 射手沿着射手→目标方向的速度分量
        shooter_angle_to_target = math.radians(bearing_to_target - shooter_hdg)
        shooter_closure = shooter_speed * math.cos(shooter_angle_to_target)

        # 目标沿着目标→射手方向的速度分量（反方向）
        bearing_to_shooter = (bearing_to_target + 180) % 360
        target_angle_to_shooter = math.radians(bearing_to_shooter - target_hdg)
        target_closure = target_speed * math.cos(target_angle_to_shooter)

        # 总接近率
        closure_rate = shooter_closure + target_closure

        return closure_rate

    @classmethod
    def calculate_nez(cls, is_manned, aspect_angle):
        """
        计算动态NEZ (No Escape Zone) - 基础版本（兼容旧接口）

        NEZ根据姿态角变化：
        - 迎头(0°): NEZ最大
        - 尾追(180°): NEZ最小
        - 线性插值中间角度
        """
        if is_manned:
            nez_head = cls.MANNED_NEZ_HEAD_ON
            nez_tail = cls.MANNED_NEZ_TAIL
        else:
            nez_head = cls.UAV_NEZ_HEAD_ON
            nez_tail = cls.UAV_NEZ_TAIL

        # 线性插值: 0° -> nez_head, 180° -> nez_tail
        nez = nez_head - (nez_head - nez_tail) * (aspect_angle / 180.0)

        return nez

    @classmethod
    def decompose_target_velocity(cls, shooter_lon, shooter_lat,
                                   target_lon, target_lat,
                                   target_vx, target_vy, target_vz=0):
        """
        分解目标速度为相对射手的径向和横向分量

        Args:
            shooter_lon, shooter_lat: 射手位置
            target_lon, target_lat: 目标位置
            target_vx: 目标速度x分量 (东为正)
            target_vy: 目标速度y分量 (北为正)
            target_vz: 目标速度z分量 (上为正)

        Returns:
            (v_radial, v_lateral): 径向速度(正=远离), 横向速度(绝对值)
        """
        # 计算射手到目标的方向向量
        dx_m = (target_lon - shooter_lon) * 111000 * math.cos(math.radians((shooter_lat + target_lat) / 2))
        dy_m = (target_lat - shooter_lat) * 111000
        dist = math.sqrt(dx_m**2 + dy_m**2)

        if dist < 1:  # 避免除零
            return 0, 0

        # 单位方向向量（从射手指向目标）
        ux, uy = dx_m / dist, dy_m / dist

        # 目标速度在射手→目标方向的投影（径向速度）
        # 正值表示目标远离射手，负值表示目标接近射手
        v_radial = target_vx * ux + target_vy * uy

        # 目标速度的横向分量（垂直于射手→目标方向）
        # 这是目标的"逃逸速度"
        target_speed_2d = math.sqrt(target_vx**2 + target_vy**2)
        v_lateral = math.sqrt(max(0, target_speed_2d**2 - v_radial**2))

        return v_radial, v_lateral

    @classmethod
    def calculate_nez_physics(cls, is_manned, target_is_manned,
                               v_radial, v_lateral, aspect_angle):
        """
        基于物理模型计算NEZ (No Escape Zone) - V2增强版

        物理原理：
        - 导弹需要在燃料耗尽前追上目标
        - 目标可以选择径向逃逸（远离）或横向逃逸（垂直机动）
        - NEZ = 导弹有效追击距离 - 目标逃逸距离

        公式推导：
        - 导弹飞行时间 t = 导弹射程 / 导弹速度
        - 目标逃逸距离 = 目标逃逸速度 × t
        - NEZ = 导弹追击距离 - 目标逃逸距离

        Args:
            is_manned: 射手是否为有人机
            target_is_manned: 目标是否为有人机
            v_radial: 目标径向速度 (正=远离射手)
            v_lateral: 目标横向速度 (逃逸速度)
            aspect_angle: 姿态角 (0=迎头, 180=尾追)

        Returns:
            nez: 动态NEZ距离 (米)
        """
        # 获取导弹和飞机参数
        if is_manned:
            missile_speed = cls.MANNED_MISSILE_SPEED
            missile_range = cls.MANNED_MISSILE_RANGE
            missile_duration = cls.MANNED_MISSILE_DURATION
        else:
            missile_speed = cls.UAV_MISSILE_SPEED
            missile_range = cls.UAV_MISSILE_RANGE
            missile_duration = cls.UAV_MISSILE_DURATION

        # 目标最大逃逸速度
        if target_is_manned:
            target_max_speed = cls.MANNED_MAX_SPEED
        else:
            target_max_speed = cls.UAV_MAX_SPEED

        # === 计算导弹相对目标的有效追击速度 ===
        # 考虑目标的径向逃逸速度
        # 如果目标远离(v_radial > 0)，导弹追击速度降低
        # 如果目标接近(v_radial < 0)，导弹追击速度提高
        effective_missile_speed = missile_speed - v_radial

        # 安全检查：如果目标逃逸速度超过导弹速度，NEZ为0
        if effective_missile_speed <= 0:
            return 0

        # === 计算导弹飞行时间 ===
        # 导弹最大飞行时间受燃料限制
        max_flight_time = missile_duration

        # === 计算横向逃逸影响 ===
        # 目标横向机动会增加导弹需要飞行的距离
        # 简化模型：假设目标以恒定横向速度逃逸
        # 导弹需要额外飞行 v_lateral × t 的距离来补偿

        # 实际逃逸速度考虑：目标可能会加速到最大速度
        # 使用当前横向速度和最大速度的较大值的一定比例
        effective_lateral = max(v_lateral, target_max_speed * 0.6)

        # === 计算NEZ ===
        # 导弹最大追击距离
        max_pursuit_distance = effective_missile_speed * max_flight_time

        # 目标横向逃逸导致的导弹额外飞行距离
        # 几何关系：导弹需要飞行 sqrt(d^2 + (v_lat*t)^2) 才能命中
        # 简化为线性惩罚因子
        lateral_penalty_factor = 1.0 + (effective_lateral / missile_speed) ** 2

        # 姿态角影响：尾追时NEZ更小（目标能量优势）
        aspect_factor = 1.0 - 0.5 * (aspect_angle / 180.0)

        # 综合NEZ计算
        nez = (max_pursuit_distance / lateral_penalty_factor) * aspect_factor

        # 限制在合理范围内
        nez = max(1000, min(nez, missile_range * 0.8))

        return nez

    @classmethod
    def calculate_coordinated_nez(cls, shooter1, shooter2, target):
        """
        计算双机协同攻击时的综合NEZ

        双机从不同方向攻击时，目标无法同时向两个方向逃逸
        综合NEZ会显著增大

        Args:
            shooter1: 射手1信息字典
            shooter2: 射手2信息字典
            target: 目标信息字典

        Returns:
            (nez1, nez2, combined_nez, escape_difficulty):
            各自NEZ和综合NEZ以及逃逸难度评分
        """
        # 获取目标速度分量
        t_vx = target.get('v_x', target.get('velocity_x', 0))
        t_vy = target.get('v_y', target.get('velocity_y', 0))
        t_vz = target.get('v_z', target.get('velocity_z', 0))
        t_lon = target.get('longitude', target.get('X', 0))
        t_lat = target.get('latitude', target.get('Y', 0))
        target_is_manned = target.get('platform_entity_type') == '有人机'

        # 射手1参数
        s1_lon = shooter1.get('longitude', 0)
        s1_lat = shooter1.get('latitude', 0)
        s1_heading = shooter1.get('heading', 0)
        s1_is_manned = shooter1.get('type') == '有人机'

        # 射手2参数
        s2_lon = shooter2.get('longitude', 0)
        s2_lat = shooter2.get('latitude', 0)
        s2_heading = shooter2.get('heading', 0)
        s2_is_manned = shooter2.get('type') == '有人机'

        # 分解目标相对于射手1的速度
        v_radial_1, v_lateral_1 = cls.decompose_target_velocity(
            s1_lon, s1_lat, t_lon, t_lat, t_vx, t_vy, t_vz
        )

        # 分解目标相对于射手2的速度
        v_radial_2, v_lateral_2 = cls.decompose_target_velocity(
            s2_lon, s2_lat, t_lon, t_lat, t_vx, t_vy, t_vz
        )

        # 计算姿态角
        aspect_1 = cls.calculate_aspect_angle(
            s1_lon, s1_lat, s1_heading,
            t_lon, t_lat, target.get('heading', 0)
        )
        aspect_2 = cls.calculate_aspect_angle(
            s2_lon, s2_lat, s2_heading,
            t_lon, t_lat, target.get('heading', 0)
        )

        # 计算各自的物理NEZ
        nez1 = cls.calculate_nez_physics(
            s1_is_manned, target_is_manned,
            v_radial_1, v_lateral_1, aspect_1
        )
        nez2 = cls.calculate_nez_physics(
            s2_is_manned, target_is_manned,
            v_radial_2, v_lateral_2, aspect_2
        )

        # === 计算逃逸难度 ===
        # 关键洞察：如果目标对一个射手有高横向速度，
        # 对另一个射手可能是低横向速度（取决于攻击角度差）

        # 攻击角度差
        angle_diff = cls._calculate_shooter_angle_diff(
            s1_lon, s1_lat, s2_lon, s2_lat, t_lon, t_lat
        )

        # 逃逸难度评分 (0-1)
        # 角度差90°时最难逃逸（正交攻击）
        # 角度差0°或180°时容易逃逸（同向或对向攻击）
        optimal_angle = 90
        angle_penalty = abs(angle_diff - optimal_angle) / 90.0
        escape_difficulty = 1.0 - angle_penalty * 0.5

        # 综合NEZ: 基于逃逸难度加权
        # 当逃逸难度高时，取较大的NEZ
        # 当逃逸难度低时，取较小的NEZ
        combined_nez = (max(nez1, nez2) * escape_difficulty +
                        min(nez1, nez2) * (1 - escape_difficulty))

        # 双机协同bonus：逃逸难度高时增大NEZ
        coordination_bonus = 1.0 + 0.3 * escape_difficulty
        combined_nez *= coordination_bonus

        return nez1, nez2, combined_nez, escape_difficulty

    @classmethod
    def _calculate_shooter_angle_diff(cls, s1_lon, s1_lat, s2_lon, s2_lat, t_lon, t_lat):
        """
        计算两架射手相对目标的攻击角度差

        返回: 0-180度
        """
        # 目标到射手1的方位角
        angle1 = math.atan2(s1_lon - t_lon, s1_lat - t_lat) * 180 / math.pi

        # 目标到射手2的方位角
        angle2 = math.atan2(s2_lon - t_lon, s2_lat - t_lat) * 180 / math.pi

        # 角度差
        diff = abs(angle1 - angle2)
        if diff > 180:
            diff = 360 - diff

        return diff

    @classmethod
    def calculate_pk(cls, distance, aspect_angle, closure_rate, is_manned, target_is_manned=False):
        """
        估算命中概率 (Pk - Kill Probability)

        v1版本：原始简单模型
        - 均匀的姿态因子分布
        - 简单的接近率判断
        - 基于NEZ的距离因子

        返回: 0.0 - 1.0 的概率值
        """
        if is_manned:
            max_range = cls.MANNED_MAX_RANGE
            optimal_range = cls.MANNED_OPTIMAL_RANGE
            nez = cls.calculate_nez(True, aspect_angle)
        else:
            max_range = cls.UAV_MAX_RANGE
            optimal_range = cls.UAV_OPTIMAL_RANGE
            nez = cls.calculate_nez(False, aspect_angle)

        # 1. 距离因子 (0.0 - 1.0)
        if distance > max_range:
            range_factor = 0.0
        elif distance <= nez:
            # NEZ内：高命中率
            range_factor = 0.90
        elif distance <= optimal_range:
            # 最优射程内：较高
            range_factor = 0.80
        else:
            # 远距离：线性衰减
            range_factor = 0.70 * (1 - (distance - optimal_range) / (max_range - optimal_range))

        # 2. 姿态因子 - v1简单版本（更均匀）
        if aspect_angle < 30:
            # 迎头：最佳
            aspect_factor = 1.0
        elif aspect_angle < 60:
            # 前侧方
            aspect_factor = 0.90
        elif aspect_angle < 90:
            # 横越
            aspect_factor = 0.85
        elif aspect_angle < 120:
            # 后侧方
            aspect_factor = 0.80
        else:
            # 尾追
            aspect_factor = 0.75

        # 3. 接近率因子 - v1简单版本
        if closure_rate < 0:
            # 远离：惩罚
            closure_factor = 0.70
        elif closure_rate < 200:
            # 慢速接近
            closure_factor = 0.90
        elif closure_rate < 400:
            # 中速接近：最佳
            closure_factor = 1.0
        else:
            # 高速接近
            closure_factor = 0.95

        # 4. 目标类型因子
        target_factor = 1.0

        # 综合Pk
        pk = range_factor * aspect_factor * closure_factor * target_factor

        # 限制在 0-1 范围
        pk = max(0.0, min(1.0, pk))

        return pk

    @classmethod
    def should_fire(cls, shooter, target, agent, debug_prefix=""):
        """
        综合判断是否应该开火

        v1版本：简单直接的Pk阈值判断
        - 无复杂的危险区检测
        - 基于Pk阈值的简单决策

        返回: (should_fire: bool, pk: float, reason: str)
        """
        # 获取射手信息
        s_lon = shooter.get('longitude', 0)
        s_lat = shooter.get('latitude', 0)
        s_speed = shooter.get('speed', 300)
        s_heading = shooter.get('heading', 0)
        is_manned = shooter.get('type') == '有人机'

        # 获取目标信息
        t_lon = target.get('longitude', target.get('X', 0))
        t_lat = target.get('latitude', target.get('Y', 0))
        t_speed = target.get('speed', 300)
        t_heading = target.get('heading', 0)
        target_is_manned = target.get('platform_entity_type') == '有人机'

        # 计算距离
        distance = YxGeoUtils.haversine_distance(s_lon, s_lat, t_lon, t_lat)

        # 最大射程检查
        max_range = cls.MANNED_MAX_RANGE if is_manned else cls.UAV_MAX_RANGE
        if distance > max_range:
            return False, 0.0, "超出射程"

        # 计算姿态角
        aspect_angle = cls.calculate_aspect_angle(
            s_lon, s_lat, s_heading,
            t_lon, t_lat, t_heading
        )

        # 计算接近率
        closure_rate = cls.calculate_closure_rate(
            s_lon, s_lat, s_speed, s_heading,
            t_lon, t_lat, t_speed, t_heading
        )

        # 计算NEZ
        nez = cls.calculate_nez(is_manned, aspect_angle)

        # 计算Pk
        pk = cls.calculate_pk(distance, aspect_angle, closure_rate, is_manned, target_is_manned)

        # === v1开火决策逻辑：简单版 ===

        # 1. 在NEZ内且Pk达标：优先开火
        if distance <= nez and pk >= cls.PK_THRESHOLD_URGENT:
            return True, pk, f"NEZ内(d={distance:.0f}m,pk={pk:.2f})"

        # 2. 高价值目标（有人机）：降低阈值
        if target_is_manned and pk >= cls.PK_THRESHOLD_HIGH_VALUE:
            return True, pk, f"高价值目标(pk={pk:.2f})"

        # 3. 正常情况：Pk达到阈值即开火
        if pk >= cls.PK_THRESHOLD_NORMAL:
            return True, pk, f"正常开火(pk={pk:.2f})"

        # 默认不开火
        return False, pk, f"Pk不足({pk:.2f}<{cls.PK_THRESHOLD_NORMAL})"

    @classmethod
    def calculate_pk_v2(cls, shooter, target, v_radial=None, v_lateral=None):
        """
        V2增强版命中概率计算 - 考虑目标逃逸速度

        Args:
            shooter: 射手信息字典
            target: 目标信息字典
            v_radial: 目标径向速度（可选，不提供则自动计算）
            v_lateral: 目标横向速度（可选，不提供则自动计算）

        Returns:
            (pk, details): 命中概率和详细信息字典
        """
        # 获取射手信息
        s_lon = shooter.get('longitude', 0)
        s_lat = shooter.get('latitude', 0)
        s_speed = shooter.get('speed', 300)
        s_heading = shooter.get('heading', 0)
        is_manned = shooter.get('type') == '有人机'

        # 获取目标信息
        t_lon = target.get('longitude', target.get('X', 0))
        t_lat = target.get('latitude', target.get('Y', 0))
        t_vx = target.get('v_x', target.get('velocity_x', 0))
        t_vy = target.get('v_y', target.get('velocity_y', 0))
        t_vz = target.get('v_z', target.get('velocity_z', 0))
        t_speed = target.get('speed', 300)
        t_heading = target.get('heading', 0)
        target_is_manned = target.get('platform_entity_type') == '有人机'

        # 计算距离
        distance = YxGeoUtils.haversine_distance(s_lon, s_lat, t_lon, t_lat)

        # 计算姿态角
        aspect_angle = cls.calculate_aspect_angle(
            s_lon, s_lat, s_heading,
            t_lon, t_lat, t_heading
        )

        # 计算速度分解（如果未提供）
        if v_radial is None or v_lateral is None:
            v_radial, v_lateral = cls.decompose_target_velocity(
                s_lon, s_lat, t_lon, t_lat, t_vx, t_vy, t_vz
            )

        # 计算物理NEZ
        nez = cls.calculate_nez_physics(
            is_manned, target_is_manned,
            v_radial, v_lateral, aspect_angle
        )

        # 获取武器参数
        if is_manned:
            max_range = cls.MANNED_MAX_RANGE
            optimal_range = cls.MANNED_OPTIMAL_RANGE
            missile_speed = cls.MANNED_MISSILE_SPEED
        else:
            max_range = cls.UAV_MAX_RANGE
            optimal_range = cls.UAV_OPTIMAL_RANGE
            missile_speed = cls.UAV_MISSILE_SPEED

        # === 1. 距离因子 (基于物理NEZ) ===
        if distance > max_range:
            range_factor = 0.0
        elif distance <= nez * 0.7:
            # 深入NEZ内：很高命中率
            range_factor = 0.95
        elif distance <= nez:
            # NEZ边缘：高命中率
            range_factor = 0.85
        elif distance <= optimal_range:
            # 最优射程内
            range_factor = 0.75
        else:
            # 远距离：线性衰减
            range_factor = 0.60 * (1 - (distance - optimal_range) / (max_range - optimal_range))

        # === 2. 姿态因子 (基于aspect angle) ===
        # 迎头最佳，尾追最差
        aspect_factor = 1.0 - 0.3 * (aspect_angle / 180.0)

        # === 3. 径向速度因子 ===
        # 目标远离会降低命中率，目标接近会提高命中率
        if v_radial > 0:
            # 远离
            radial_penalty = min(v_radial / missile_speed, 0.3)
            radial_factor = 1.0 - radial_penalty
        else:
            # 接近
            radial_bonus = min(abs(v_radial) / missile_speed, 0.15)
            radial_factor = 1.0 + radial_bonus

        # === 4. 横向速度因子 (逃逸惩罚) ===
        # 高横向速度 = 高逃逸能力 = 低命中率
        target_max_speed = cls.MANNED_MAX_SPEED if target_is_manned else cls.UAV_MAX_SPEED
        lateral_ratio = v_lateral / target_max_speed
        lateral_factor = 1.0 - 0.4 * min(lateral_ratio, 1.0)

        # === 5. 目标类型因子 ===
        # 有人机更大，更容易被命中
        target_factor = 1.1 if target_is_manned else 1.0

        # 综合Pk
        pk = range_factor * aspect_factor * radial_factor * lateral_factor * target_factor
        pk = max(0.0, min(1.0, pk))

        # 详细信息
        details = {
            'distance': distance,
            'nez': nez,
            'aspect_angle': aspect_angle,
            'v_radial': v_radial,
            'v_lateral': v_lateral,
            'range_factor': range_factor,
            'aspect_factor': aspect_factor,
            'radial_factor': radial_factor,
            'lateral_factor': lateral_factor,
            'in_nez': distance <= nez
        }

        return pk, details

    @classmethod
    def should_coordinated_fire(cls, shooter1, shooter2, target, agent=None):
        """
        判断双机是否应该协同开火

        考虑因素：
        1. 两架飞机各自的Pk
        2. 双机攻击角度差（正交攻击最优）
        3. 目标相对两架飞机的逃逸速度分析
        4. 综合NEZ判断

        Returns:
            (should_fire, combined_pk, nez_info, reason)
        """
        # 获取目标信息
        t_lon = target.get('longitude', target.get('X', 0))
        t_lat = target.get('latitude', target.get('Y', 0))
        t_vx = target.get('v_x', target.get('velocity_x', 0))
        t_vy = target.get('v_y', target.get('velocity_y', 0))
        target_is_manned = target.get('platform_entity_type') == '有人机'

        # 获取射手位置
        s1_lon = shooter1.get('longitude', 0)
        s1_lat = shooter1.get('latitude', 0)
        s1_is_manned = shooter1.get('type') == '有人机'

        s2_lon = shooter2.get('longitude', 0)
        s2_lat = shooter2.get('latitude', 0)
        s2_is_manned = shooter2.get('type') == '有人机'

        # 计算距离
        dist1 = YxGeoUtils.haversine_distance(s1_lon, s1_lat, t_lon, t_lat)
        dist2 = YxGeoUtils.haversine_distance(s2_lon, s2_lat, t_lon, t_lat)

        # 检查射程
        max_range_1 = cls.MANNED_MAX_RANGE if s1_is_manned else cls.UAV_MAX_RANGE
        max_range_2 = cls.MANNED_MAX_RANGE if s2_is_manned else cls.UAV_MAX_RANGE

        if dist1 > max_range_1 or dist2 > max_range_2:
            return False, 0.0, {}, "超出射程"

        # 计算协同NEZ
        nez1, nez2, combined_nez, escape_difficulty = cls.calculate_coordinated_nez(
            shooter1, shooter2, target
        )

        # 分解目标速度
        v_radial_1, v_lateral_1 = cls.decompose_target_velocity(
            s1_lon, s1_lat, t_lon, t_lat, t_vx, t_vy
        )
        v_radial_2, v_lateral_2 = cls.decompose_target_velocity(
            s2_lon, s2_lat, t_lon, t_lat, t_vx, t_vy
        )

        # 计算各自Pk
        pk1, details1 = cls.calculate_pk_v2(shooter1, target, v_radial_1, v_lateral_1)
        pk2, details2 = cls.calculate_pk_v2(shooter2, target, v_radial_2, v_lateral_2)

        # === 协同开火Pk计算 ===
        # 双发导弹的综合命中率: 1 - (1-pk1)*(1-pk2)
        combined_pk = 1.0 - (1.0 - pk1) * (1.0 - pk2)

        # 逃逸难度加成
        # 当攻击角度差大时，目标难以同时规避两个方向
        coordination_bonus = 1.0 + 0.2 * escape_difficulty
        combined_pk = min(1.0, combined_pk * coordination_bonus)

        # NEZ信息
        nez_info = {
            'nez1': nez1,
            'nez2': nez2,
            'combined_nez': combined_nez,
            'escape_difficulty': escape_difficulty,
            'dist1': dist1,
            'dist2': dist2,
            'pk1': pk1,
            'pk2': pk2,
            'v_radial_1': v_radial_1,
            'v_lateral_1': v_lateral_1,
            'v_radial_2': v_radial_2,
            'v_lateral_2': v_lateral_2,
            'in_nez1': dist1 <= nez1,
            'in_nez2': dist2 <= nez2,
            'in_combined_nez': max(dist1, dist2) <= combined_nez
        }

        # === 开火决策 ===

        # 1. 双机都在NEZ内 - 最佳条件
        if dist1 <= nez1 and dist2 <= nez2:
            if combined_pk >= cls.PK_THRESHOLD_COORDINATED:
                return True, combined_pk, nez_info, f"双机NEZ内(pk={combined_pk:.2f},逃逸难度={escape_difficulty:.2f})"

        # 2. 在综合NEZ内且逃逸难度高
        if max(dist1, dist2) <= combined_nez and escape_difficulty >= 0.6:
            if combined_pk >= cls.PK_THRESHOLD_COORDINATED:
                return True, combined_pk, nez_info, f"协同NEZ内(pk={combined_pk:.2f},逃逸难度={escape_difficulty:.2f})"

        # 3. 高价值目标 - 降低阈值
        if target_is_manned and combined_pk >= cls.PK_THRESHOLD_HIGH_VALUE:
            return True, combined_pk, nez_info, f"高价值目标(pk={combined_pk:.2f})"

        # 4. 综合Pk足够高
        if combined_pk >= cls.PK_THRESHOLD_NORMAL:
            return True, combined_pk, nez_info, f"协同开火(pk={combined_pk:.2f})"

        # 默认不开火
        return False, combined_pk, nez_info, f"Pk不足({combined_pk:.2f}<{cls.PK_THRESHOLD_COORDINATED})"
