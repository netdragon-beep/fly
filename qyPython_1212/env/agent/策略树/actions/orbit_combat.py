"""
空中盘旋作战动作节点

实现有人机和无人机在空中盘旋与敌机作战的战术

战术设计原则：
1. 当发现敌机时，不直接冲向敌机，而是采用盘旋战术
2. 有人机在外围较安全位置盘旋，保持雷达锁定
3. 无人机在更近的距离盘旋，形成包围圈
4. 盘旋过程中寻找最佳射击角度
5. 利用多机协同，从不同方向发起攻击

盘旋参数基于官方参数：
- 有人机：速度180-500m/s，高度2000-7000m，雷达60km
- 无人机：速度120-360m/s，高度2000-7000m，雷达40km
- 导弹：速度1200m/s，最大射程72km，有效射程35km
"""

import math
from ..bt_framework import Action, NodeStatus
from ..fire_control import SmartFireControl
from utilities.yxScriptTreeFunc import YxScriptTreeFunc as decCmd
from utilities.yxGeoUtils import YxGeoUtils


class OfficialParams:
    """官方赛题参数"""
    # 有人机参数
    MANNED_MIN_SPEED = 180
    MANNED_MAX_SPEED = 500
    MANNED_MIN_ALT = 2000
    MANNED_MAX_ALT = 7000
    MANNED_RADAR_RANGE = 60000  # 60km

    # 无人机参数
    UAV_MIN_SPEED = 120
    UAV_MAX_SPEED = 360
    UAV_MIN_ALT = 2000
    UAV_MAX_ALT = 7000
    UAV_RADAR_RANGE = 40000  # 40km

    # 导弹参数
    MISSILE_SPEED = 1200
    MISSILE_MAX_RANGE = 72000
    MISSILE_EFFECTIVE_RANGE = 35000  # 有效射程

    # 经纬度转换
    KM_PER_DEGREE = 111.0
    DEG_PER_KM = 1.0 / 111.0

    @classmethod
    def km_to_deg(cls, km):
        return km * cls.DEG_PER_KM

    @classmethod
    def m_to_deg(cls, m):
        return m / 1000.0 * cls.DEG_PER_KM


class ActionOrbitCombat(Action):
    """
    空中盘旋作战

    战术设计：
    1. 以敌机群为中心，有人机和无人机在不同半径上盘旋
    2. 有人机在外围（安全距离），保持雷达锁定，提供态势感知
    3. 无人机在内圈（攻击距离），随时准备发起攻击
    4. 盘旋方向根据双方态势动态调整
    5. 多机在不同相位盘旋，形成全方位包围

    盘旋参数：
    - 有人机盘旋半径：30-35km（在雷达范围内但保持安全距离）
    - 无人机盘旋半径：20-25km（在有效射程边缘）
    - 盘旋速度：有人机280-300m/s，无人机300-320m/s
    - 盘旋高度：根据敌机高度动态调整，保持高度优势
    """

    # === 盘旋参数配置 ===
    # 有人机盘旋参数（安全距离，主要提供态势感知）
    MANNED_ORBIT_RADIUS_KM = 32          # 有人机盘旋半径 km
    MANNED_ORBIT_SPEED = 280             # 有人机盘旋速度 m/s
    MANNED_ORBIT_ALT_OFFSET = 500        # 有人机相对敌机高度偏移 m（保持高度优势）

    # 无人机盘旋参数（攻击距离，准备发起攻击）
    UAV_ORBIT_RADIUS_KM = 22             # 无人机盘旋半径 km（在有效射程边缘）
    UAV_ORBIT_SPEED = 310                # 无人机盘旋速度 m/s
    UAV_ORBIT_ALT_OFFSET = 200           # 无人机相对敌机高度偏移 m

    # 盘旋角速度（度/帧）
    ORBIT_ANGULAR_SPEED_MANNED = 2.0     # 有人机每帧转过的角度
    ORBIT_ANGULAR_SPEED_UAV = 2.5        # 无人机每帧转过的角度

    # 高度限制
    MIN_ALTITUDE = 2000
    MAX_ALTITUDE = 7000

    # 触发条件
    MIN_ENEMY_DISTANCE = 50000           # 最小触发距离（敌机距离小于此值才盘旋）
    MAX_ENEMY_DISTANCE = 100000          # 最大触发距离（超过此值执行接近机动而非盘旋）

    # 安全距离
    MANNED_SAFE_DISTANCE = 25000         # 有人机与敌机的最小安全距离

    # 边界安全边距（经纬度，约10km）
    BOUNDARY_MARGIN = 0.09

    def _clamp_to_boundary(self, lon, lat, agent):
        """将坐标限制在战场边界内"""
        bf = agent.battlefield
        if bf.get('min_lon') is not None:
            lon = max(bf['min_lon'] + self.BOUNDARY_MARGIN,
                     min(bf['max_lon'] - self.BOUNDARY_MARGIN, lon))
            lat = max(bf['min_lat'] + self.BOUNDARY_MARGIN,
                     min(bf['max_lat'] - self.BOUNDARY_MARGIN, lat))
        return lon, lat

    def tick(self, agent) -> str:
        """执行盘旋作战逻辑"""

        # 没有敌机时不执行
        if not agent.enemy_units:
            return NodeStatus.SUCCESS

        # 获取可用单位
        manned = [u for u in agent.own_units if u.get('type') == '有人机'
                  and u['name'] not in agent.commanded_units]
        uavs = [u for u in agent.own_units if u.get('type') == '无人机'
                and u['name'] not in agent.commanded_units]

        if not manned and not uavs:
            return NodeStatus.SUCCESS

        # 计算敌机群中心（优先以敌方有人机为中心）
        enemy_center = self._calculate_enemy_center(agent.enemy_units)
        if enemy_center is None:
            return NodeStatus.SUCCESS

        enemy_center_lat, enemy_center_lon, enemy_center_alt = enemy_center

        # 检查是否在盘旋作战触发范围内
        all_units = manned + uavs
        avg_dist = self._calculate_average_distance_to_point(
            all_units, enemy_center_lon, enemy_center_lat
        )

        # 距离过远时执行接近机动（而非盘旋）
        if avg_dist > self.MAX_ENEMY_DISTANCE:
            self._execute_approach_maneuver(agent, manned, uavs, enemy_center_lat, enemy_center_lon, enemy_center_alt)
            return NodeStatus.SUCCESS

        # 初始化盘旋角度
        if not hasattr(agent, 'orbit_combat_angle'):
            agent.orbit_combat_angle = 0
        agent.orbit_combat_angle = (agent.orbit_combat_angle + self.ORBIT_ANGULAR_SPEED_UAV) % 360

        # === 有人机外围盘旋 ===
        if manned:
            self._execute_manned_orbit(
                agent, manned, enemy_center_lat, enemy_center_lon, enemy_center_alt
            )

        # === 无人机内圈盘旋 ===
        if uavs:
            self._execute_uav_orbit(
                agent, uavs, enemy_center_lat, enemy_center_lon, enemy_center_alt
            )

        return NodeStatus.SUCCESS

    def _calculate_enemy_center(self, enemy_units):
        """
        计算敌机群中心

        优先以敌方有人机为中心（高价值目标）
        如果没有有人机，则计算所有敌机的几何中心
        """
        if not enemy_units:
            return None

        # 优先以敌方有人机为中心
        enemy_manned = [e for e in enemy_units if e.get('platform_entity_type') == '有人机']
        if enemy_manned:
            center_units = enemy_manned
        else:
            center_units = enemy_units

        total_lat = 0
        total_lon = 0
        total_alt = 0
        count = 0

        for enemy in center_units:
            lat = enemy.get('latitude', enemy.get('Y'))
            lon = enemy.get('longitude', enemy.get('X'))
            alt = enemy.get('altitude', enemy.get('Alt', 4000))

            if lat is not None and lon is not None:
                total_lat += lat
                total_lon += lon
                total_alt += alt
                count += 1

        if count == 0:
            return None

        return (total_lat / count, total_lon / count, total_alt / count)

    def _calculate_average_distance_to_point(self, units, target_lon, target_lat):
        """计算单位到目标点的平均距离"""
        if not units:
            return float('inf')

        total_dist = 0
        count = 0

        for unit in units:
            u_lon = unit.get('longitude', unit.get('X'))
            u_lat = unit.get('latitude', unit.get('Y'))
            if u_lon is not None and u_lat is not None:
                dist = YxGeoUtils.haversine_distance(u_lon, u_lat, target_lon, target_lat)
                total_dist += dist
                count += 1

        return total_dist / count if count > 0 else float('inf')

    def _execute_manned_orbit(self, agent, manned_units, enemy_lat, enemy_lon, enemy_alt):
        """
        执行有人机外围盘旋

        有人机在较远距离盘旋，保持安全的同时提供态势感知
        """
        orbit_radius_deg = OfficialParams.km_to_deg(self.MANNED_ORBIT_RADIUS_KM)

        for i, unit in enumerate(manned_units):
            # 检查与敌机的距离，确保不会太近
            u_lon = unit.get('longitude', unit.get('X', 0))
            u_lat = unit.get('latitude', unit.get('Y', 0))

            dist_to_enemy = YxGeoUtils.haversine_distance(u_lon, u_lat, enemy_lon, enemy_lat)

            # 如果太近，先拉开距离
            if dist_to_enemy < self.MANNED_SAFE_DISTANCE:
                self._retreat_from_enemy(agent, unit, enemy_lon, enemy_lat, enemy_alt)
                continue

            # 计算盘旋位置
            # 多架有人机均匀分布在圆周上
            num_units = len(manned_units)
            angle_offset = (360.0 / num_units) * i
            angle = (agent.orbit_combat_angle * 0.8 + angle_offset) % 360  # 有人机转得慢一点
            angle_rad = math.radians(angle)

            target_lat = enemy_lat + orbit_radius_deg * math.sin(angle_rad)
            target_lon = enemy_lon + orbit_radius_deg * math.cos(angle_rad)

            # 边界检查
            target_lon, target_lat = self._clamp_to_boundary(target_lon, target_lat, agent)

            # 计算目标高度（保持高度优势）
            target_alt = enemy_alt + self.MANNED_ORBIT_ALT_OFFSET
            target_alt = max(self.MIN_ALTITUDE, min(self.MAX_ALTITUDE, target_alt))

            target_pt = (target_lat, target_lon, target_alt)
            agent.add_action(
                decCmd.fly_to_point(unit['name'], target_pt, self.MANNED_ORBIT_SPEED),
                unit['name']
            )

    def _execute_uav_orbit(self, agent, uav_units, enemy_lat, enemy_lon, enemy_alt):
        """
        执行无人机内圈盘旋

        无人机在攻击距离盘旋，准备发起攻击
        """
        orbit_radius_deg = OfficialParams.km_to_deg(self.UAV_ORBIT_RADIUS_KM)

        for i, unit in enumerate(uav_units):
            # 检查是否有弹药，无弹药的无人机不参与进攻盘旋
            has_ammo = False
            for weapon in unit.get('weapons', []):
                if weapon.get('quantity', 0) > 0:
                    has_ammo = True
                    break

            if not has_ammo:
                # 无弹药，跳过（让保护有人机的逻辑处理）
                continue

            # 计算盘旋位置
            # 多架无人机均匀分布在圆周上，与有人机相位错开
            num_units = len(uav_units)
            angle_offset = (360.0 / num_units) * i + 45  # 与有人机错开45度
            angle = (agent.orbit_combat_angle + angle_offset) % 360
            angle_rad = math.radians(angle)

            target_lat = enemy_lat + orbit_radius_deg * math.sin(angle_rad)
            target_lon = enemy_lon + orbit_radius_deg * math.cos(angle_rad)

            # 边界检查
            target_lon, target_lat = self._clamp_to_boundary(target_lon, target_lat, agent)

            # 计算目标高度
            target_alt = enemy_alt + self.UAV_ORBIT_ALT_OFFSET
            target_alt = max(self.MIN_ALTITUDE, min(self.MAX_ALTITUDE, target_alt))

            target_pt = (target_lat, target_lon, target_alt)
            agent.add_action(
                decCmd.fly_to_point(unit['name'], target_pt, self.UAV_ORBIT_SPEED),
                unit['name']
            )

    def _retreat_from_enemy(self, agent, unit, enemy_lon, enemy_lat, enemy_alt):
        """
        有人机后撤

        当有人机与敌机距离太近时，先拉开距离
        """
        u_lon = unit.get('longitude', unit.get('X', 0))
        u_lat = unit.get('latitude', unit.get('Y', 0))
        u_alt = unit.get('altitude', unit.get('Alt', 4000))

        # 计算从敌机指向我方的方向（即后撤方向）
        direction = YxGeoUtils.calculate_direction_to(enemy_lon, enemy_lat, u_lon, u_lat)

        # 后撤目标距离
        retreat_dist_km = self.MANNED_ORBIT_RADIUS_KM + 5  # 后撤到盘旋半径外
        lon_off, lat_off = YxGeoUtils.km_to_lon_lat(agent.center_lat, retreat_dist_km, direction)

        target_lat = enemy_lat + lat_off
        target_lon = enemy_lon + lon_off

        # 边界检查
        target_lon, target_lat = self._clamp_to_boundary(target_lon, target_lat, agent)

        target_alt = max(self.MIN_ALTITUDE, min(self.MAX_ALTITUDE, u_alt + 500))

        target_pt = (target_lat, target_lon, target_alt)

        # 使用最大速度后撤
        agent.add_action(
            decCmd.fly_to_point(unit['name'], target_pt, OfficialParams.MANNED_MAX_SPEED),
            unit['name']
        )

    def _execute_approach_maneuver(self, agent, manned_units, uav_units, enemy_lat, enemy_lon, enemy_alt):
        """
        执行接近机动

        当敌机距离过远时（>100km），执行接近机动而非盘旋
        无人机在前，有人机在后，向敌机方向推进
        """
        # 无人机快速接近
        for unit in uav_units:
            # 检查弹药
            has_ammo = any(w.get('quantity', 0) > 0 for w in unit.get('weapons', []))
            if not has_ammo:
                continue  # 无弹药的让保护有人机逻辑处理

            u_lon = unit.get('longitude', unit.get('X', 0))
            u_lat = unit.get('latitude', unit.get('Y', 0))

            # 计算目标位置：敌机前方一定距离
            direction = YxGeoUtils.calculate_direction_to(u_lon, u_lat, enemy_lon, enemy_lat)
            # 目标是到达盘旋半径位置
            approach_dist_km = self.UAV_ORBIT_RADIUS_KM + 10  # 比盘旋半径远一点
            lon_off, lat_off = YxGeoUtils.km_to_lon_lat(agent.center_lat, approach_dist_km, direction)

            target_lat = enemy_lat - lat_off  # 从敌机方向反向偏移
            target_lon = enemy_lon - lon_off

            # 边界检查
            target_lon, target_lat = self._clamp_to_boundary(target_lon, target_lat, agent)

            target_alt = max(self.MIN_ALTITUDE, min(self.MAX_ALTITUDE, enemy_alt + 200))

            target_pt = (target_lat, target_lon, target_alt)
            agent.add_action(
                decCmd.fly_to_point(unit['name'], target_pt, OfficialParams.UAV_MAX_SPEED),
                unit['name']
            )

        # 有人机稍慢跟进，保持在后方
        for unit in manned_units:
            u_lon = unit.get('longitude', unit.get('X', 0))
            u_lat = unit.get('latitude', unit.get('Y', 0))

            # 有人机目标位置：比无人机更远
            direction = YxGeoUtils.calculate_direction_to(u_lon, u_lat, enemy_lon, enemy_lat)
            approach_dist_km = self.MANNED_ORBIT_RADIUS_KM + 15  # 保持更远的安全距离
            lon_off, lat_off = YxGeoUtils.km_to_lon_lat(agent.center_lat, approach_dist_km, direction)

            target_lat = enemy_lat - lat_off
            target_lon = enemy_lon - lon_off

            # 边界检查
            target_lon, target_lat = self._clamp_to_boundary(target_lon, target_lat, agent)

            target_alt = max(self.MIN_ALTITUDE, min(self.MAX_ALTITUDE, enemy_alt + 500))

            target_pt = (target_lat, target_lon, target_alt)
            # 有人机速度稍慢，让无人机在前
            agent.add_action(
                decCmd.fly_to_point(unit['name'], target_pt, 400),
                unit['name']
            )


class ActionAggressiveOrbitCombat(Action):
    """
    激进盘旋作战

    与普通盘旋作战的区别：
    1. 盘旋半径更小（更接近敌机）
    2. 无人机会主动收缩包围圈
    3. 适合在己方数量优势时使用
    """

    # 更激进的盘旋参数
    MANNED_ORBIT_RADIUS_KM = 28          # 有人机盘旋半径（更近）
    UAV_ORBIT_RADIUS_KM = 18             # 无人机盘旋半径（更近，在最优射程内）

    UAV_ORBIT_SPEED = 330                # 无人机更快
    ORBIT_ANGULAR_SPEED = 3.0            # 更快的角速度

    MIN_ALTITUDE = 2000
    MAX_ALTITUDE = 7000

    # 边界安全边距（经纬度，约10km）
    BOUNDARY_MARGIN = 0.09

    def _clamp_to_boundary(self, lon, lat, agent):
        """将坐标限制在战场边界内"""
        bf = agent.battlefield
        if bf.get('min_lon') is not None:
            lon = max(bf['min_lon'] + self.BOUNDARY_MARGIN,
                     min(bf['max_lon'] - self.BOUNDARY_MARGIN, lon))
            lat = max(bf['min_lat'] + self.BOUNDARY_MARGIN,
                     min(bf['max_lat'] - self.BOUNDARY_MARGIN, lat))
        return lon, lat

    def tick(self, agent) -> str:
        """执行激进盘旋作战"""

        if not agent.enemy_units:
            return NodeStatus.SUCCESS

        manned = [u for u in agent.own_units if u.get('type') == '有人机'
                  and u['name'] not in agent.commanded_units]
        uavs = [u for u in agent.own_units if u.get('type') == '无人机'
                and u['name'] not in agent.commanded_units]

        if not manned and not uavs:
            return NodeStatus.SUCCESS

        # 计算敌机群中心
        enemy_center = self._calculate_enemy_center(agent.enemy_units)
        if enemy_center is None:
            return NodeStatus.SUCCESS

        enemy_lat, enemy_lon, enemy_alt = enemy_center

        # 初始化角度
        if not hasattr(agent, 'aggressive_orbit_angle'):
            agent.aggressive_orbit_angle = 0
        agent.aggressive_orbit_angle = (agent.aggressive_orbit_angle + self.ORBIT_ANGULAR_SPEED) % 360

        # 有人机盘旋
        if manned:
            orbit_radius_deg = OfficialParams.km_to_deg(self.MANNED_ORBIT_RADIUS_KM)
            for i, unit in enumerate(manned):
                num_units = len(manned)
                angle = (agent.aggressive_orbit_angle * 0.7 + (360.0 / num_units) * i) % 360
                angle_rad = math.radians(angle)

                target_lat = enemy_lat + orbit_radius_deg * math.sin(angle_rad)
                target_lon = enemy_lon + orbit_radius_deg * math.cos(angle_rad)

                # 边界检查
                target_lon, target_lat = self._clamp_to_boundary(target_lon, target_lat, agent)

                target_alt = max(self.MIN_ALTITUDE, min(self.MAX_ALTITUDE, enemy_alt + 400))

                agent.add_action(
                    decCmd.fly_to_point(unit['name'], (target_lat, target_lon, target_alt), 300),
                    unit['name']
                )

        # 无人机收缩盘旋
        if uavs:
            orbit_radius_deg = OfficialParams.km_to_deg(self.UAV_ORBIT_RADIUS_KM)
            for i, unit in enumerate(uavs):
                # 检查弹药
                has_ammo = any(w.get('quantity', 0) > 0 for w in unit.get('weapons', []))
                if not has_ammo:
                    continue

                num_units = len(uavs)
                angle = (agent.aggressive_orbit_angle + (360.0 / num_units) * i + 30) % 360
                angle_rad = math.radians(angle)

                target_lat = enemy_lat + orbit_radius_deg * math.sin(angle_rad)
                target_lon = enemy_lon + orbit_radius_deg * math.cos(angle_rad)

                # 边界检查
                target_lon, target_lat = self._clamp_to_boundary(target_lon, target_lat, agent)

                target_alt = max(self.MIN_ALTITUDE, min(self.MAX_ALTITUDE, enemy_alt + 200))

                agent.add_action(
                    decCmd.fly_to_point(unit['name'], (target_lat, target_lon, target_alt), self.UAV_ORBIT_SPEED),
                    unit['name']
                )

        return NodeStatus.SUCCESS

    def _calculate_enemy_center(self, enemy_units):
        """计算敌机群中心"""
        if not enemy_units:
            return None

        enemy_manned = [e for e in enemy_units if e.get('platform_entity_type') == '有人机']
        center_units = enemy_manned if enemy_manned else enemy_units

        total_lat = sum(e.get('latitude', e.get('Y', 0)) for e in center_units)
        total_lon = sum(e.get('longitude', e.get('X', 0)) for e in center_units)
        total_alt = sum(e.get('altitude', e.get('Alt', 4000)) for e in center_units)
        count = len(center_units)

        if count == 0:
            return None

        return (total_lat / count, total_lon / count, total_alt / count)


class ActionDefensiveOrbitCombat(Action):
    """
    防御盘旋作战

    保守战术：
    1. 有人机保持在非常安全的距离
    2. 无人机在有人机和敌机之间形成屏障
    3. 适合在己方有人机需要保护时使用
    """

    # 防御性盘旋参数
    MANNED_ORBIT_RADIUS_KM = 40          # 有人机盘旋半径（更远，更安全）
    UAV_ORBIT_RADIUS_KM = 28             # 无人机盘旋半径（在有人机和敌机之间）

    MANNED_ORBIT_SPEED = 260             # 有人机低速节能
    UAV_ORBIT_SPEED = 300                # 无人机警戒速度

    MIN_ALTITUDE = 2000
    MAX_ALTITUDE = 7000

    # 边界安全边距（经纬度，约10km）
    BOUNDARY_MARGIN = 0.09

    def _clamp_to_boundary(self, lon, lat, agent):
        """将坐标限制在战场边界内"""
        bf = agent.battlefield
        if bf.get('min_lon') is not None:
            lon = max(bf['min_lon'] + self.BOUNDARY_MARGIN,
                     min(bf['max_lon'] - self.BOUNDARY_MARGIN, lon))
            lat = max(bf['min_lat'] + self.BOUNDARY_MARGIN,
                     min(bf['max_lat'] - self.BOUNDARY_MARGIN, lat))
        return lon, lat

    def tick(self, agent) -> str:
        """执行防御盘旋作战"""

        if not agent.enemy_units:
            return NodeStatus.SUCCESS

        manned = [u for u in agent.own_units if u.get('type') == '有人机'
                  and u['name'] not in agent.commanded_units]
        uavs = [u for u in agent.own_units if u.get('type') == '无人机'
                and u['name'] not in agent.commanded_units]

        if not manned and not uavs:
            return NodeStatus.SUCCESS

        # 计算敌机群中心
        enemy_center = self._calculate_enemy_center(agent.enemy_units)
        if enemy_center is None:
            return NodeStatus.SUCCESS

        enemy_lat, enemy_lon, enemy_alt = enemy_center

        # 初始化角度
        if not hasattr(agent, 'defensive_orbit_angle'):
            agent.defensive_orbit_angle = 0
        agent.defensive_orbit_angle = (agent.defensive_orbit_angle + 1.5) % 360

        # 有人机远距离盘旋
        if manned:
            orbit_radius_deg = OfficialParams.km_to_deg(self.MANNED_ORBIT_RADIUS_KM)
            for i, unit in enumerate(manned):
                num_units = len(manned)
                angle = (agent.defensive_orbit_angle * 0.5 + (360.0 / num_units) * i) % 360
                angle_rad = math.radians(angle)

                target_lat = enemy_lat + orbit_radius_deg * math.sin(angle_rad)
                target_lon = enemy_lon + orbit_radius_deg * math.cos(angle_rad)

                # 边界检查
                target_lon, target_lat = self._clamp_to_boundary(target_lon, target_lat, agent)

                target_alt = max(self.MIN_ALTITUDE, min(self.MAX_ALTITUDE, enemy_alt + 800))

                agent.add_action(
                    decCmd.fly_to_point(unit['name'], (target_lat, target_lon, target_alt), self.MANNED_ORBIT_SPEED),
                    unit['name']
                )

        # 无人机中距离屏障
        if uavs:
            orbit_radius_deg = OfficialParams.km_to_deg(self.UAV_ORBIT_RADIUS_KM)
            for i, unit in enumerate(uavs):
                num_units = len(uavs)
                # 无人机在有人机前方（朝向敌机方向）
                angle = (agent.defensive_orbit_angle + (360.0 / num_units) * i) % 360
                angle_rad = math.radians(angle)

                target_lat = enemy_lat + orbit_radius_deg * math.sin(angle_rad)
                target_lon = enemy_lon + orbit_radius_deg * math.cos(angle_rad)

                # 边界检查
                target_lon, target_lat = self._clamp_to_boundary(target_lon, target_lat, agent)

                target_alt = max(self.MIN_ALTITUDE, min(self.MAX_ALTITUDE, enemy_alt + 300))

                agent.add_action(
                    decCmd.fly_to_point(unit['name'], (target_lat, target_lon, target_alt), self.UAV_ORBIT_SPEED),
                    unit['name']
                )

        return NodeStatus.SUCCESS

    def _calculate_enemy_center(self, enemy_units):
        """计算敌机群中心"""
        if not enemy_units:
            return None

        enemy_manned = [e for e in enemy_units if e.get('platform_entity_type') == '有人机']
        center_units = enemy_manned if enemy_manned else enemy_units

        total_lat = sum(e.get('latitude', e.get('Y', 0)) for e in center_units)
        total_lon = sum(e.get('longitude', e.get('X', 0)) for e in center_units)
        total_alt = sum(e.get('altitude', e.get('Alt', 4000)) for e in center_units)
        count = len(center_units)

        if count == 0:
            return None

        return (total_lat / count, total_lon / count, total_alt / count)
