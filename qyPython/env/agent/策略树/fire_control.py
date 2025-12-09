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

    # === 开火阈值 ===
    PK_THRESHOLD_NORMAL = 0.5      # 正常情况下的Pk阈值
    PK_THRESHOLD_URGENT = 0.35     # 紧急情况（目标逃跑）的Pk阈值
    PK_THRESHOLD_HIGH_VALUE = 0.4  # 高价值目标（有人机）的Pk阈值

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

        影响因素：
        1. 距离因子 - 越近越好，但有最优距离
        2. 姿态因子 - 迎头最佳，尾追最差
        3. 接近率因子 - 接近时更好
        4. 目标类型因子 - 有人机更大更容易命中

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
            # 在NEZ内，高命中率
            range_factor = 0.9 + 0.1 * (1 - distance / nez)
        elif distance <= optimal_range:
            # 在最优范围内
            range_factor = 0.7 + 0.2 * (1 - (distance - nez) / (optimal_range - nez))
        else:
            # 最优范围到最大射程之间，命中率快速下降
            range_factor = 0.7 * (1 - (distance - optimal_range) / (max_range - optimal_range)) ** 2

        # 2. 姿态因子 (0.3 - 1.0)
        # 迎头(0°) = 1.0, 尾追(180°) = 0.3
        aspect_factor = 1.0 - 0.7 * (aspect_angle / 180.0)

        # 3. 接近率因子 (0.5 - 1.2)
        # 快速接近加成，远离惩罚
        if closure_rate > 200:  # 快速接近 (>200 m/s)
            closure_factor = 1.2
        elif closure_rate > 0:  # 缓慢接近
            closure_factor = 1.0 + 0.2 * (closure_rate / 200)
        elif closure_rate > -100:  # 缓慢远离
            closure_factor = 0.8 + 0.2 * (1 + closure_rate / 100)
        else:  # 快速远离
            closure_factor = 0.5

        # 4. 目标类型因子
        # 有人机目标更大，稍微容易命中
        target_factor = 1.1 if target_is_manned else 1.0

        # 综合Pk
        pk = range_factor * aspect_factor * closure_factor * target_factor

        # 限制在 0-1 范围
        pk = max(0.0, min(1.0, pk))

        return pk

    @classmethod
    def should_fire(cls, shooter, target, agent, debug_prefix=""):
        """
        综合判断是否应该开火

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

        # === 开火决策逻辑 ===

        # 情况1: 在NEZ内，高优先级开火
        if distance <= nez:
            if pk >= cls.PK_THRESHOLD_URGENT:
                return True, pk, f"NEZ内(d={distance:.0f}m)"

        # 情况2: 目标是高价值目标（有人机）
        if target_is_manned:
            if pk >= cls.PK_THRESHOLD_HIGH_VALUE:
                return True, pk, f"高价值目标(Pk={pk:.2f})"

        # 情况3: 目标正在逃跑且有一定命中率
        if closure_rate < -50:  # 目标在逃跑
            if pk >= cls.PK_THRESHOLD_URGENT:
                return True, pk, f"目标逃跑(cr={closure_rate:.0f})"

        # 情况4: 正常情况，Pk达到阈值
        if pk >= cls.PK_THRESHOLD_NORMAL:
            return True, pk, f"正常开火(Pk={pk:.2f})"

        # 情况5: 目标正在接近，等待更好时机
        if closure_rate > 100:
            return False, pk, f"等待接近(cr={closure_rate:.0f})"

        # 默认不开火
        return False, pk, f"Pk不足({pk:.2f}<{cls.PK_THRESHOLD_NORMAL})"
