"""
智能火控系统 (Smart Fire Control)

核心概念：
1. WEZ (Weapon Engagement Zone) - 武器可攻击区，最大射程内
2. NEZ (No Escape Zone) - 不可逃逸区，目标无法规避的范围
3. Pk (Kill Probability) - 命中概率估算

算法原理：
- 不在最大射程边缘开火，而是等待进入最优攻击区
- 考虑目标的接近率(closure rate)和姿态角(aspect angle)
- 只有当Pk超过阈值时才开火
"""

import math
from utilities.yxGeoUtils import YxGeoUtils


class SmartFireControl:
    """
    智能火控系统 - 提升导弹命中率
    """

    # === 武器参数 ===
    # 有人机武器 (远程导弹)
    MANNED_MAX_RANGE = 20000       # 最大射程 20km
    MANNED_NEZ_HEAD_ON = 12000     # 迎头NEZ 12km
    MANNED_NEZ_TAIL = 6000         # 尾追NEZ 6km
    MANNED_OPTIMAL_RANGE = 15000   # 最优射程 15km
    MANNED_MISSILE_SPEED = 900     # 导弹速度 m/s

    # 无人机武器 (近程导弹)
    UAV_MAX_RANGE = 15000          # 最大射程 15km
    UAV_NEZ_HEAD_ON = 8000         # 迎头NEZ 8km
    UAV_NEZ_TAIL = 4000            # 尾追NEZ 4km
    UAV_OPTIMAL_RANGE = 10000      # 最优射程 10km
    UAV_MISSILE_SPEED = 800        # 导弹速度 m/s

    # === 开火阈值 (v1: 原始版本) ===
    PK_THRESHOLD_NORMAL = 0.45     # 正常情况下的Pk阈值
    PK_THRESHOLD_URGENT = 0.40     # 紧急情况的Pk阈值
    PK_THRESHOLD_HIGH_VALUE = 0.40 # 高价值目标（有人机）的Pk阈值

    # === 姿态角限制 ===
    MAX_ASPECT_FOR_NEZ_FIRE = 70   # NEZ内开火最大姿态角（超过70°不建议开火）
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
        计算动态NEZ (No Escape Zone)

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
