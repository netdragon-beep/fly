"""
阵型和战术动作节点

包含：
- ActionSearchFormation: 分散搜索阵型
- ActionMannedRetreat: 有人机后撤
- ActionProtectMannedVision: 保护有人机视野
- ActionCenterPatrol: 中心巡逻
- ActionPatrolFormation: 防御/巡逻阵型
"""

import math
from ..bt_framework import Action, NodeStatus
from utilities.yxScriptTreeFunc import YxScriptTreeFunc as decCmd
from utilities.yxGeoUtils import YxGeoUtils


class ActionSearchFormation(Action):
    """
    分散搜索阵型：根据侦查范围散开，持续向前推进

    战术思路：
    - 无人机和有人机并排，根据侦查范围横向散开
    - 持续向敌方方向推进，直到经过中心30km
    - 只有经过中心30km后仍无敌机才停止搜索
    """
    # 侦查范围配置（km）
    UAV_DETECTION_RANGE = 15
    MANNED_DETECTION_RANGE = 25
    ADVANCE_STEP = 0.06  # 每帧推进距离（约6km）
    PASS_CENTER_THRESHOLD = 0.27  # 经过中心30km（约0.27度）

    def tick(self, agent) -> str:
        # 如果已发现敌机，不执行搜索阵型
        if agent.enemy_units:
            return NodeStatus.SUCCESS

        manned = [u for u in agent.own_units if u.get('type') == '有人机'
                  and u['name'] not in agent.commanded_units]
        uavs = [u for u in agent.own_units if u.get('type') == '无人机'
                and u['name'] not in agent.commanded_units]

        if not manned and not uavs:
            return NodeStatus.SUCCESS

        center_lat = agent.center_lat
        center_lon = agent.center_lon
        enemy_dir = 1 if agent.side == 'red' else -1

        # 计算当前阵线位置
        all_units = manned + uavs
        avg_lon = sum(u.get('longitude', 0) for u in all_units) / len(all_units)

        # 检查是否已经过中心30km（此时应该开始盘旋，由ActionCenterPatrol处理）
        passed_center_dist = (avg_lon - center_lon) * enemy_dir
        if passed_center_dist > self.PASS_CENTER_THRESHOLD:
            return NodeStatus.SUCCESS  # 交给ActionCenterPatrol处理

        # === 继续向前推进搜索 ===
        # 目标搜索线：当前位置向前推进
        search_lon = avg_lon + (self.ADVANCE_STEP * enemy_dir)

        # 所有飞机并排展开
        all_sorted = sorted(all_units, key=lambda u: u.get('latitude', 0))
        num_units = len(all_sorted)
        total_width_deg = 0.35
        start_lat = center_lat - total_width_deg / 2

        for i, unit in enumerate(all_sorted):
            target_lat = start_lat + (i + 0.5) * (total_width_deg / num_units)
            target_lon = search_lon

            if unit.get('type') == '有人机':
                target_lon = search_lon + (0.02 * enemy_dir)
                speed = 350
                alt = 4000
            else:
                speed = 400
                alt = 3500

            target_pt = (target_lat, target_lon, alt)
            agent.add_action(decCmd.fly_to_point(unit['name'], target_pt, speed), unit['name'])

        return NodeStatus.SUCCESS


class ActionMannedRetreat(Action):
    """
    发现敌机时：有人机减速后撤盘旋
    """
    def tick(self, agent) -> str:
        # 只有发现敌机时才执行
        if not agent.enemy_units:
            return NodeStatus.SUCCESS

        manned = [u for u in agent.own_units if u.get('type') == '有人机'
                  and u['name'] not in agent.commanded_units]

        if not manned:
            return NodeStatus.SUCCESS

        center_lat = agent.center_lat
        center_lon = agent.center_lon
        enemy_dir = 1 if agent.side == 'red' else -1

        # 有人机后撤位置（己方一侧）
        retreat_lon = center_lon + (-0.2 * enemy_dir)

        # 盘旋角度
        if not hasattr(agent, 'manned_retreat_angle'):
            agent.manned_retreat_angle = 0
        agent.manned_retreat_angle = (agent.manned_retreat_angle + 8) % 360

        for i, unit in enumerate(manned):
            # 在后撤位置小范围盘旋
            angle = (agent.manned_retreat_angle + i * 180) % 360
            angle_rad = math.radians(angle)

            retreat_radius = 0.05  # 约5km小圆盘旋
            target_lat = center_lat + retreat_radius * math.cos(angle_rad)
            target_lon = retreat_lon + retreat_radius * math.sin(angle_rad)

            target_pt = (target_lat, target_lon, 4500)  # 高度提升
            agent.add_action(decCmd.fly_to_point(unit['name'], target_pt, 280), unit['name'])  # 减速

        return NodeStatus.SUCCESS


class ActionProtectMannedVision(Action):
    """
    无弹药的无人机：保护有人机，为有人机提供视野盲区覆盖

    战术思路：
    - 没有导弹的无人机不再追击
    - 在有人机周围巡逻，覆盖有人机的视野盲区（后方和侧方）
    - 同时帮助占领中心区域
    """
    def tick(self, agent) -> str:
        # 找出没有弹药的无人机
        uavs_no_ammo = []
        for uav in agent.own_units:
            if uav.get('type') != '无人机':
                continue
            if uav['name'] in agent.commanded_units:
                continue
            # 检查是否有弹药
            has_ammo = False
            for weapon in uav.get('weapons', []):
                if weapon.get('quantity', 0) > 0:
                    has_ammo = True
                    break
            if not has_ammo:
                uavs_no_ammo.append(uav)

        if not uavs_no_ammo:
            return NodeStatus.SUCCESS

        # 找有人机位置
        manned = [u for u in agent.own_units if u.get('type') == '有人机']

        if manned:
            # 有人机存在，围绕有人机保护
            avg_m_lat = sum(m.get('latitude', 0) for m in manned) / len(manned)
            avg_m_lon = sum(m.get('longitude', 0) for m in manned) / len(manned)
            protect_center_lat = avg_m_lat
            protect_center_lon = avg_m_lon
        else:
            # 无有人机，在中心区域巡逻
            protect_center_lat = agent.center_lat
            protect_center_lon = agent.center_lon

        # 无弹药无人机围绕保护点巡逻（覆盖视野盲区）
        if not hasattr(agent, 'protect_angle'):
            agent.protect_angle = 0
        agent.protect_angle = (agent.protect_angle + 6) % 360

        enemy_dir = 1 if agent.side == 'red' else -1
        num_protect = len(uavs_no_ammo)

        for i, uav in enumerate(uavs_no_ammo):
            # 在有人机后方和侧方分布（覆盖盲区）
            # 盲区主要在后方（己方方向）和两侧
            # 角度分布：从后方120度到240度（即后半圆）
            base_angle = 180  # 后方
            spread = 120  # 覆盖范围
            if num_protect > 1:
                unit_angle = base_angle - spread/2 + (i / (num_protect - 1)) * spread
            else:
                unit_angle = base_angle

            # 转换为绝对角度（考虑敌方方向）
            if enemy_dir > 0:  # 红方，敌人在东，后方是西
                abs_angle = (270 + unit_angle) % 360
            else:  # 蓝方，敌人在西，后方是东
                abs_angle = (90 + unit_angle) % 360

            # 加上旋转偏移，形成巡逻效果
            final_angle = (abs_angle + agent.protect_angle * 0.3) % 360
            angle_rad = math.radians(final_angle)

            # 保护半径约8km
            protect_radius = 0.07
            target_lat = protect_center_lat + protect_radius * math.cos(angle_rad)
            target_lon = protect_center_lon + protect_radius * math.sin(angle_rad)

            target_pt = (target_lat, target_lon, 3800)
            agent.add_action(decCmd.fly_to_point(uav['name'], target_pt, 350), uav['name'])

        return NodeStatus.SUCCESS


class ActionCenterPatrol(Action):
    """
    经过中心30km且无敌机：在地图中心点盘旋巡逻
    """
    # 经过中心30km后才开始盘旋
    PASS_CENTER_THRESHOLD = 0.27  # 约30km

    def tick(self, agent) -> str:
        # 如果有敌机，不执行盘旋
        if agent.enemy_units:
            return NodeStatus.SUCCESS

        manned = [u for u in agent.own_units if u.get('type') == '有人机'
                  and u['name'] not in agent.commanded_units]
        uavs = [u for u in agent.own_units if u.get('type') == '无人机'
                and u['name'] not in agent.commanded_units]

        if not manned and not uavs:
            return NodeStatus.SUCCESS

        # === 从战场数据计算真实中心点 ===
        battlefield = agent.battlefield
        if battlefield.get('min_lon') is not None and battlefield.get('max_lon') is not None:
            real_center_lon = (battlefield['min_lon'] + battlefield['max_lon']) / 2
            real_center_lat = (battlefield['min_lat'] + battlefield['max_lat']) / 2
        else:
            # 备用默认值
            real_center_lon = agent.center_lon
            real_center_lat = agent.center_lat

        # 调试输出（只输出一次）
        if not hasattr(agent, '_patrol_center_printed'):
            agent._patrol_center_printed = True
            print(f"[ActionCenterPatrol] 战场边界: min_lon={battlefield.get('min_lon')}, max_lon={battlefield.get('max_lon')}, "
                  f"min_lat={battlefield.get('min_lat')}, max_lat={battlefield.get('max_lat')}")
            print(f"[ActionCenterPatrol] 计算的中心点: lat={real_center_lat}, lon={real_center_lon}")

        # 检查是否已经过中心30km
        all_units = manned + uavs
        avg_lon = sum(u.get('longitude', 0) for u in all_units) / len(all_units)
        enemy_dir = 1 if agent.side == 'red' else -1
        passed_center_dist = (avg_lon - real_center_lon) * enemy_dir

        if passed_center_dist < self.PASS_CENTER_THRESHOLD:
            return NodeStatus.SUCCESS  # 还没经过中心30km，继续搜索

        # === 已搜索完毕（经过中心30km无敌机），返回中心点盘旋 ===
        # 盘旋圆心是地图中心点，不是当前位置
        patrol_center_lat = real_center_lat
        patrol_center_lon = real_center_lon

        # 标记进入盘旋模式
        if not hasattr(agent, '_entered_patrol_mode'):
            agent._entered_patrol_mode = True
            print(f"[ActionCenterPatrol] 搜索完毕，返回中心点盘旋: lat={patrol_center_lat}, lon={patrol_center_lon}")

        if not hasattr(agent, 'center_patrol_angle'):
            agent.center_patrol_angle = 0
        agent.center_patrol_angle = (agent.center_patrol_angle + 5) % 360

        # 有人机内环（围绕地图中心点）
        if manned:
            inner_radius = 0.06  # 约6km
            for i, unit in enumerate(manned):
                angle = (agent.center_patrol_angle + i * (360 / len(manned))) % 360
                angle_rad = math.radians(angle)
                target_lat = patrol_center_lat + inner_radius * math.cos(angle_rad)
                target_lon = patrol_center_lon + inner_radius * math.sin(angle_rad)
                target_pt = (target_lat, target_lon, 4000)
                agent.add_action(decCmd.fly_to_point(unit['name'], target_pt, 350), unit['name'])

        # 无人机外环（围绕地图中心点）
        if uavs:
            outer_radius = 0.15  # 约15km
            for i, uav in enumerate(uavs):
                angle = (agent.center_patrol_angle + i * (360 / len(uavs))) % 360
                angle_rad = math.radians(angle)
                target_lat = patrol_center_lat + outer_radius * math.cos(angle_rad)
                target_lon = patrol_center_lon + outer_radius * math.sin(angle_rad)
                target_pt = (target_lat, target_lon, 3500)
                agent.add_action(decCmd.fly_to_point(uav['name'], target_pt, 400), uav['name'])

        return NodeStatus.SUCCESS


class ActionPatrolFormation(Action):
    """防御/巡逻阵型"""
    def tick(self, agent) -> str:
        # 仅控制剩下的单位
        available_units = [u for u in agent.own_units if u['name'] not in agent.commanded_units]
        if not available_units:
            return NodeStatus.SUCCESS

        manned = [u for u in available_units if u.get('type') == '有人机']
        uavs = [u for u in available_units if u.get('type') == '无人机']

        # 内环有人机
        if manned:  # 防止除零
            for i, unit in enumerate(manned):
                angle = (i / len(manned) * 360 + agent.defense_angle_offset) % 360
                lon_off, lat_off = YxGeoUtils.km_to_lon_lat(agent.center_lat, 15, angle)
                pt = (agent.center_lat + lat_off, agent.center_lon + lon_off, 4000)
                agent.add_action(decCmd.fly_to_point(unit['name'], pt, 300), unit['name'])

        # 外环无人机
        if uavs:  # 防止除零
            for i, unit in enumerate(uavs):
                angle = (i / len(uavs) * 360 + agent.defense_angle_offset) % 360
                lon_off, lat_off = YxGeoUtils.km_to_lon_lat(agent.center_lat, 30, angle)
                pt = (agent.center_lat + lat_off, agent.center_lon + lon_off, 3500)
                agent.add_action(decCmd.fly_to_point(unit['name'], pt, 300), unit['name'])

        return NodeStatus.SUCCESS
