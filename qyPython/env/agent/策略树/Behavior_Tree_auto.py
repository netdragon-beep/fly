import math
import random
import config
from typing import List, Dict, Tuple, Set
from env.agent.agent_base import AutoAgentBase
from utilities.yxScriptTreeFunc import YxScriptTreeFunc as decCmd
from utilities.yxGeoUtils import YxGeoUtils

# ==========================================
# Part 1: 轻量级行为树框架 (Mini BT Engine)
# ==========================================

class BTNode:
    """行为树节点基类"""
    def tick(self, agent) -> str:
        raise NotImplementedError

class NodeStatus:
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"

class Sequence(BTNode):
    """序列节点 (AND)：所有子节点成功才算成功，按顺序执行"""
    def __init__(self, children: List[BTNode]):
        self.children = children

    def tick(self, agent) -> str:
        for child in self.children:
            status = child.tick(agent)
            if status != NodeStatus.SUCCESS:
                return status
        return NodeStatus.SUCCESS

class Selector(BTNode):
    """选择节点 (OR)：只要有一个子节点成功就算成功"""
    def __init__(self, children: List[BTNode]):
        self.children = children

    def tick(self, agent) -> str:
        for child in self.children:
            status = child.tick(agent)
            if status == NodeStatus.SUCCESS:
                return NodeStatus.SUCCESS
            if status == NodeStatus.RUNNING:
                return NodeStatus.RUNNING
        return NodeStatus.FAILURE

class Action(BTNode):
    """动作节点基类"""
    pass

class Condition(BTNode):
    """条件节点基类"""
    pass

# ==========================================
# Part 2: 具体的业务逻辑节点 (Leaf Nodes)
# ==========================================

class ActionResetFrame(Action):
    """每一帧开始前的清理工作"""
    def tick(self, agent) -> str:
        agent.current_actions = []        # 清空指令列表
        agent.commanded_units = set()     # 清空已被占用的单位集合
        agent.missile_threats = []        # 清空威胁缓存
        return NodeStatus.SUCCESS

class ConditionCheckInitialDeployment(Condition):
    """检查是否完成了初始部署"""
    def tick(self, agent) -> str:
        if not agent.initial_deployment_complete:
            return NodeStatus.SUCCESS # 需要部署
        return NodeStatus.FAILURE     # 不需要部署

class ActionExecuteDeployment(Action):
    """执行开局部署"""
    def tick(self, agent) -> str:
        # 分离有人机和无人机
        manned = [u for u in agent.own_units if u.get('type') == '有人机']
        uavs = [u for u in agent.own_units if u.get('type') == '无人机']
        
        # 无人机靠前
        for unit in uavs:
            offset_lon = 0.45 if agent.side == 'red' else -0.45
            target = (unit['latitude'], unit['longitude'] + offset_lon, 3500)
            agent.add_action(decCmd.fly_to_point(unit['name'], target, 400), unit['name'])
            
        # 有人机靠后
        for unit in manned:
            offset_lon = 0.27 if agent.side == 'red' else -0.27
            target = (unit['latitude'], unit['longitude'] + offset_lon, 4000)
            agent.add_action(decCmd.fly_to_point(unit['name'], target, 300), unit['name'])

        agent.initial_deployment_complete = True
        return NodeStatus.SUCCESS

class ActionEvadeMissiles(Action):
    """
    多导弹协同规避 - Cranking机动 + 高度规避

    核心改进（解决导弹追踪问题）：
    1. Cranking机动：不是垂直逃跑，而是斜向后方飞行（约120-150°角度）
       - 既增加横向距离，又增加纵向距离
       - 持续让导弹需要转向追踪，消耗导弹能量
    2. 高度骤降：紧急时快速下降，利用高度差增加导弹追踪难度
    3. 更早规避：35km就开始规避，给自己更多反应时间
    4. 更大规避距离：规避点设置得更远

    导弹参数（假设）：
    - 导弹速度：1200 m/s
    - 飞机最大速度：900 m/s
    - 导弹能量有限，持续转向会消耗能量导致失速
    """
    # === 威胁判断参数 ===
    MISSILE_THREAT_RADIUS = 35000   # 威胁半径35km（更早规避）
    MISSILE_CRITICAL_RADIUS = 12000 # 紧急规避半径12km
    MISSILE_DANGER_RADIUS = 20000   # 危险半径20km
    MISSILE_THREAT_ANGLE = 90       # 导弹威胁角度（收紧到90度，减少误判）

    # === 规避行为参数 ===
    EVADE_SPEED_NORMAL = 800        # 正常规避速度
    EVADE_SPEED_CRITICAL = 900      # 紧急规避速度（最大）
    EVADE_DISTANCE_NORMAL = 20      # 正常规避距离（km）- 增大到20km
    EVADE_DISTANCE_CRITICAL = 25    # 紧急规避距离（km）- 增大到25km

    # === Cranking机动参数 ===
    CRANK_ANGLE_NORMAL = 120        # 正常Crank角度（向后斜飞120°）
    CRANK_ANGLE_CRITICAL = 150      # 紧急Crank角度（更向后，150°）

    # === 高度规避参数 ===
    ALTITUDE_DROP_NORMAL = 800      # 正常下降高度
    ALTITUDE_DROP_CRITICAL = 1500   # 紧急下降高度（大幅下降）
    MIN_ALTITUDE = 500              # 最低高度限制
    MAX_ALTITUDE = 6000             # 最高高度限制

    # === 边界安全 ===
    BOUNDARY_MARGIN = 0.12          # 边界安全边距（约13km）

    # === 调试 ===
    DEBUG_ENABLED = True
    DEBUG_INTERVAL = 20

    def tick(self, agent) -> str:
        # 调试输出
        if self.DEBUG_ENABLED:
            if not hasattr(agent, '_missile_debug_frame'):
                agent._missile_debug_frame = 0
            if agent.frame_count - agent._missile_debug_frame >= self.DEBUG_INTERVAL:
                agent._missile_debug_frame = agent.frame_count
                print(f"[导弹规避] Frame {agent.frame_count}: 检测到 {len(agent.enemy_missiles)} 枚敌方导弹")

        if not agent.enemy_missiles:
            return NodeStatus.SUCCESS

        # 收集导弹信息
        all_missiles = []
        for missile in agent.enemy_missiles:
            m_lon = missile.get('longitude', 0)
            m_lat = missile.get('latitude', 0)
            m_heading = math.degrees(missile.get('heading', 0)) % 360
            m_speed = missile.get('speed', 1200)
            if m_lon and m_lat:
                all_missiles.append({
                    'lon': m_lon, 'lat': m_lat,
                    'heading': m_heading, 'speed': m_speed,
                    'id': missile.get('target_id') or missile.get('name') or id(missile)
                })

        if not all_missiles:
            return NodeStatus.SUCCESS

        # 调试：输出导弹信息
        if self.DEBUG_ENABLED and agent.frame_count - agent._missile_debug_frame < 3:
            for i, m in enumerate(all_missiles):
                print(f"  导弹{i+1}: ({m['lat']:.4f}, {m['lon']:.4f}), 航向={m['heading']:.1f}°")

        evade_count = 0
        for unit in agent.own_units:
            unit_name = unit['name']
            u_lon = unit.get('longitude', 0)
            u_lat = unit.get('latitude', 0)
            u_alt = unit.get('altitude', 3000)

            if not u_lon or not u_lat:
                continue

            # === 分析威胁导弹 ===
            threat_missiles = []
            min_dist = float('inf')
            primary_threat = None  # 最近的威胁导弹

            for missile in all_missiles:
                dist = YxGeoUtils.haversine_distance(u_lon, u_lat, missile['lon'], missile['lat'])

                if dist > self.MISSILE_THREAT_RADIUS:
                    continue

                # 检查是否在导弹飞行方向上
                bearing_to_unit = YxGeoUtils.calculate_bearing(
                    missile['lon'], missile['lat'], u_lon, u_lat
                )
                angle_diff = abs((bearing_to_unit - missile['heading'] + 180) % 360 - 180)

                # 调试输出
                if self.DEBUG_ENABLED and agent.frame_count - agent._missile_debug_frame < 3:
                    print(f"    {unit_name}: 距导弹{dist:.0f}m, 角度差={angle_diff:.1f}°")

                if angle_diff < self.MISSILE_THREAT_ANGLE:
                    threat_missiles.append({
                        **missile,
                        'dist': dist,
                        'bearing_to_unit': bearing_to_unit
                    })
                    if dist < min_dist:
                        min_dist = dist
                        primary_threat = threat_missiles[-1]

            # === 没有威胁，跳过 ===
            if not threat_missiles:
                continue

            # === 确定威胁等级 ===
            is_critical = min_dist < self.MISSILE_CRITICAL_RADIUS
            is_danger = min_dist < self.MISSILE_DANGER_RADIUS

            if is_critical:
                evade_speed = self.EVADE_SPEED_CRITICAL
                evade_dist = self.EVADE_DISTANCE_CRITICAL
                crank_angle = self.CRANK_ANGLE_CRITICAL
                alt_drop = self.ALTITUDE_DROP_CRITICAL
                mode_str = "[紧急]"
            elif is_danger:
                evade_speed = self.EVADE_SPEED_CRITICAL
                evade_dist = self.EVADE_DISTANCE_NORMAL
                crank_angle = self.CRANK_ANGLE_NORMAL
                alt_drop = self.ALTITUDE_DROP_NORMAL
                mode_str = "[危险]"
            else:
                evade_speed = self.EVADE_SPEED_NORMAL
                evade_dist = self.EVADE_DISTANCE_NORMAL
                crank_angle = self.CRANK_ANGLE_NORMAL
                alt_drop = self.ALTITUDE_DROP_NORMAL // 2
                mode_str = "[正常]"

            # === 计算 Cranking 规避方向 ===
            # Cranking: 向导弹来向的斜后方飞行
            # 导弹航向 -> 我要飞向导弹的反方向再偏转crank_angle
            missile_heading = primary_threat['heading']

            # 反向（背对导弹）
            away_dir = (missile_heading + 180) % 360

            # 选择左Crank还是右Crank（选择离边界远的方向）
            left_crank = (away_dir - (180 - crank_angle)) % 360
            right_crank = (away_dir + (180 - crank_angle)) % 360

            # 计算两个方向的目标点
            left_lon_off, left_lat_off = YxGeoUtils.km_to_lon_lat(u_lat, evade_dist, left_crank)
            right_lon_off, right_lat_off = YxGeoUtils.km_to_lon_lat(u_lat, evade_dist, right_crank)

            left_target_lon = u_lon + left_lon_off
            left_target_lat = u_lat + left_lat_off
            right_target_lon = u_lon + right_lon_off
            right_target_lat = u_lat + right_lat_off

            # 检查边界安全性
            left_safe = self._check_boundary_safe(left_target_lon, left_target_lat, agent)
            right_safe = self._check_boundary_safe(right_target_lon, right_target_lat, agent)

            # 选择规避方向
            if left_safe and not right_safe:
                evade_dir = left_crank
            elif right_safe and not left_safe:
                evade_dir = right_crank
            else:
                # 两边都安全或都不安全，选择离战场中心更近的
                left_dist_center = abs(left_target_lat - agent.center_lat) + abs(left_target_lon - agent.center_lon)
                right_dist_center = abs(right_target_lat - agent.center_lat) + abs(right_target_lon - agent.center_lon)
                evade_dir = left_crank if left_dist_center < right_dist_center else right_crank

            # === 计算规避目标点 ===
            lon_off, lat_off = YxGeoUtils.km_to_lon_lat(u_lat, evade_dist, evade_dir)
            evade_lon = u_lon + lon_off
            evade_lat = u_lat + lat_off

            # === 高度规避：紧急时下降 ===
            # 下降可以增加导弹的俯角追踪难度，同时利用地面杂波
            if is_critical:
                # 紧急下降
                evade_alt = max(u_alt - alt_drop, self.MIN_ALTITUDE)
            elif is_danger:
                # 交替上升/下降
                if unit_name.endswith(('1', '3')):
                    evade_alt = max(u_alt - alt_drop, self.MIN_ALTITUDE)
                else:
                    evade_alt = min(u_alt + alt_drop // 2, self.MAX_ALTITUDE)
            else:
                evade_alt = u_alt  # 保持高度

            # === 执行规避 ===
            agent.add_action(
                decCmd.fly_to_point(unit_name, (evade_lat, evade_lon, evade_alt), evade_speed),
                unit_name
            )
            evade_count += 1

            # 调试输出
            if self.DEBUG_ENABLED:
                if is_critical:
                    print(f"    [紧急] {unit_name} 距离={min_dist:.0f}m")
                print(f"[规避执行] {mode_str} {unit_name} 规避 {len(threat_missiles)} 枚导弹，"
                      f"最近距离={min_dist:.0f}m, Crank={crank_angle}°, "
                      f"目标=({evade_lat:.4f}, {evade_lon:.4f}, {evade_alt:.0f}m)")

        if self.DEBUG_ENABLED and evade_count > 0:
            print(f"[导弹规避] Frame {agent.frame_count}: {evade_count} 个单位执行Cranking规避")

        return NodeStatus.SUCCESS

    def _check_boundary_safe(self, lon, lat, agent):
        """检查坐标是否在安全边界内"""
        bf = agent.battlefield
        if bf.get('min_lon') is None:
            return True
        margin = self.BOUNDARY_MARGIN
        return (bf['min_lon'] + margin < lon < bf['max_lon'] - margin and
                bf['min_lat'] + margin < lat < bf['max_lat'] - margin)


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


# ==========================================
# Part 2.5: 智能火控系统 (Smart Fire Control)
# ==========================================

class SmartFireControl:
    """
    智能火控系统 - 提升导弹命中率

    核心概念：
    1. WEZ (Weapon Engagement Zone) - 武器可攻击区，最大射程内
    2. NEZ (No Escape Zone) - 不可逃逸区，目标无法规避的范围
    3. Pk (Kill Probability) - 命中概率估算

    算法原理：
    - 不在最大射程边缘开火，而是等待进入最优攻击区
    - 考虑目标的接近率(closure rate)和姿态角(aspect angle)
    - 只有当Pk超过阈值时才开火
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


class ActionAttackLogic(Action):
    """
    攻击逻辑：包含智能火控和机动

    采用 Shoot-Look-Shoot (发射-观察-再发射) 策略：
    - 对同一目标发射导弹后，等待导弹到达（命中或脱靶）
    - 只有确认结果后才考虑发射第二颗导弹
    - 避免浪费弹药的齐射行为
    """

    # 调试开关
    DEBUG_ENABLED = True
    DEBUG_INTERVAL = 25  # 每N帧输出一次火控信息

    # Shoot-Look-Shoot 参数
    MISSILE_FLIGHT_TIME_ESTIMATE = 80  # 估计导弹飞行时间（帧），约8秒@10fps
    MIN_REFIRE_INTERVAL = 40           # 最小再次发射间隔（帧），约4秒

    def tick(self, agent) -> str:
        if not agent.enemy_units:
            return NodeStatus.SUCCESS

        # === 初始化 Shoot-Look-Shoot 追踪器 ===
        if not hasattr(agent, 'pending_missiles'):
            agent.pending_missiles = {}  # {(shooter_name, target_id): launch_frame}

        # === 清理已过期的待定导弹记录 ===
        # 如果发射时间已超过预计飞行时间，认为导弹已到达（命中或脱靶）
        expired_keys = []
        for key, launch_frame in agent.pending_missiles.items():
            if agent.frame_count - launch_frame > self.MISSILE_FLIGHT_TIME_ESTIMATE:
                expired_keys.append(key)
        for key in expired_keys:
            del agent.pending_missiles[key]
            if self.DEBUG_ENABLED:
                print(f"[Shoot-Look-Shoot] 导弹追踪过期: {key[0]} -> 目标{key[1]}")

        # === 检查己方导弹状态（如果有的话）===
        # 通过检测目标是否还存在来判断导弹是否命中
        current_enemy_ids = set()
        for enemy in agent.enemy_units:
            target_id = enemy.get('target_id', enemy.get('id', id(enemy)))
            current_enemy_ids.add(target_id)

        # 如果目标已不存在（被击落），清除相关的pending记录
        keys_to_remove = []
        for key in agent.pending_missiles.keys():
            shooter_name, target_id = key
            if target_id not in current_enemy_ids:
                keys_to_remove.append(key)
                if self.DEBUG_ENABLED:
                    print(f"[Shoot-Look-Shoot] 目标{target_id}已被击落，{shooter_name}可再次开火")
        for key in keys_to_remove:
            del agent.pending_missiles[key]

        # 调试帧计数
        if not hasattr(agent, '_fire_control_debug_frame'):
            agent._fire_control_debug_frame = 0
        should_debug = (agent.frame_count - agent._fire_control_debug_frame >= self.DEBUG_INTERVAL)
        if should_debug:
            agent._fire_control_debug_frame = agent.frame_count

        # Sub-step 1: 智能火控 (Smart Fire Control)
        # 收集所有可能的射击方案，使用SmartFireControl评估
        fire_candidates = []
        for unit in agent.own_units:
            # 检查是否有弹药
            has_ammo = False
            for weapon in unit.get('weapons', []):
                if weapon.get('quantity', 0) > 0:
                    has_ammo = True
                    break
            if not has_ammo:
                continue

            for enemy in agent.enemy_units:
                # 安全获取坐标
                u_lon = unit.get('longitude', unit.get('X', 0))
                u_lat = unit.get('latitude', unit.get('Y', 0))
                e_lon = enemy.get('longitude', enemy.get('X', 0))
                e_lat = enemy.get('latitude', enemy.get('Y', 0))

                if u_lon is None or u_lat is None or e_lon is None or e_lat is None:
                    continue

                # 使用智能火控系统评估
                should_fire, pk, reason = SmartFireControl.should_fire(unit, enemy, agent)

                # 计算距离用于排序
                d = YxGeoUtils.haversine_distance(u_lon, u_lat, e_lon, e_lat)

                # 判断敌机类型：有人机优先级更高
                enemy_type = enemy.get('platform_entity_type', '无人机')
                priority = 0 if enemy_type == '有人机' else 1  # 0=高优先级

                fire_candidates.append({
                    'unit': unit,
                    'enemy': enemy,
                    'dist': d,
                    'priority': priority,
                    'should_fire': should_fire,
                    'pk': pk,
                    'reason': reason
                })

        # 按优先级和Pk排序（优先高价值目标，其次高Pk）
        fire_candidates.sort(key=lambda x: (x['priority'], -x['pk']))

        cur_round_shots = {}  # 本轮发射记录
        fired_units = set()   # 本轮已开火的单位

        # 调试输出
        if self.DEBUG_ENABLED and should_debug and fire_candidates:
            print(f"\n[智能火控] Frame {agent.frame_count}: 评估 {len(fire_candidates)} 个射击方案")
            print(f"  待定导弹数: {len(agent.pending_missiles)}")
            # 输出前5个最佳方案
            for i, cand in enumerate(fire_candidates[:5]):
                unit_name = cand['unit']['name']
                enemy_name = cand['enemy'].get('target_name', cand['enemy'].get('name', '?'))
                print(f"  方案{i+1}: {unit_name} -> {enemy_name}, "
                      f"Pk={cand['pk']:.2f}, 距离={cand['dist']:.0f}m, "
                      f"决策={cand['should_fire']}, 原因={cand['reason']}")

        for cand in fire_candidates:
            u, e = cand['unit'], cand['enemy']

            # 每个单位每帧只开火一次
            if u['name'] in fired_units:
                continue

            # 智能火控判断不应该开火
            if not cand['should_fire']:
                continue

            # 安全获取 target_id
            target_id = e.get('target_id', e.get('id', id(e)))

            # === Shoot-Look-Shoot 检查 ===
            # 检查该射手是否已有导弹正在飞向该目标
            pending_key = (u['name'], target_id)
            if pending_key in agent.pending_missiles:
                launch_frame = agent.pending_missiles[pending_key]
                frames_elapsed = agent.frame_count - launch_frame
                # 如果还没到最小再发射间隔，跳过
                if frames_elapsed < self.MIN_REFIRE_INTERVAL:
                    if self.DEBUG_ENABLED and should_debug:
                        print(f"  [Shoot-Look-Shoot] {u['name']} 等待上一枚导弹结果 "
                              f"(已过{frames_elapsed}帧/{self.MISSILE_FLIGHT_TIME_ESTIMATE}帧)")
                    continue

            # 敌机未被过度攻击 (最多4发导弹)
            already_fired = e.get('is_fired_num', 0) + cur_round_shots.get(target_id, 0)
            if already_fired >= 4:
                continue

            # 安全获取 target_name
            target_name = e.get('target_name', e.get('name', ''))
            if not target_name:
                continue

            # 发射！
            if agent.try_fire_weapon(u):
                agent.add_action(decCmd.fire_track(u['name'], target_name), None)
                cur_round_shots[target_id] = cur_round_shots.get(target_id, 0) + 1
                fired_units.add(u['name'])

                # === 记录 Shoot-Look-Shoot 追踪 ===
                agent.pending_missiles[pending_key] = agent.frame_count

                if self.DEBUG_ENABLED:
                    print(f"[开火] {u['name']} -> {target_name}, Pk={cand['pk']:.2f}, "
                          f"距离={cand['dist']:.0f}m, 原因={cand['reason']}")

        # Sub-step 2: Maneuver to Attack (仅对未被占用的单位有效)
        for unit in agent.own_units:
            if unit['name'] in agent.commanded_units:
                continue  # 正在规避的单位不执行进攻机动

            u_lon = unit.get('longitude', unit.get('X'))
            u_lat = unit.get('latitude', unit.get('Y'))
            if u_lon is None or u_lat is None:
                continue

            # 找最近敌机（优先有人机）
            closest = None
            min_d = float('inf')
            best_priority = 999

            for enemy in agent.enemy_units:
                e_lon = enemy.get('longitude', enemy.get('X'))
                e_lat = enemy.get('latitude', enemy.get('Y'))
                if e_lon is None or e_lat is None:
                    continue

                d = YxGeoUtils.haversine_distance(u_lon, u_lat, e_lon, e_lat)
                enemy_type = enemy.get('platform_entity_type', '无人机')
                priority = 0 if enemy_type == '有人机' else 1

                # 优先级更高 或 同优先级但更近
                if priority < best_priority or (priority == best_priority and d < min_d):
                    min_d = d
                    closest = enemy
                    best_priority = priority

            if closest:
                e_lon = closest.get('longitude', closest.get('X', 0))
                e_lat = closest.get('latitude', closest.get('Y', 0))
                e_alt = closest.get('altitude', closest.get('Alt', 3000))

                # 使用智能火控的最优攻击距离
                is_manned = unit.get('type') == '有人机'
                optimal_dist = SmartFireControl.MANNED_OPTIMAL_RANGE if is_manned else SmartFireControl.UAV_OPTIMAL_RANGE

                # 如果已经在最优距离内，不需要继续接近
                if min_d <= optimal_dist:
                    # 维持当前位置或轻微调整
                    continue

                # 飞向攻击占位点（最优射程位置）
                direction = YxGeoUtils.calculate_direction_to(e_lon, e_lat, u_lon, u_lat)
                lon_off, lat_off = YxGeoUtils.km_to_lon_lat(agent.center_lat, optimal_dist / 1000, direction)

                target_pt = (e_lat + lat_off, e_lon + lon_off, e_alt)
                # 速度根据距离调整：远的快接近，近的慢接近
                approach_speed = 550 if min_d > optimal_dist * 1.5 else 450
                agent.add_action(decCmd.fly_to_point(unit['name'], target_pt, approach_speed), unit['name'])

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

# ==========================================
# Part 3: 主智能体类 (Agent)
# ==========================================

class BTDemoAgent(AutoAgentBase):
    def __init__(self, side, name):
        super().__init__(side, name)

        # === 调试：打印战场信息 ===
        print(f"[BTDemoAgent] 初始化 side={side}, name={name}")
        print(f"[BTDemoAgent] 战场边界: {self.battlefield}")
        if self.battlefield.get('min_lon') is not None:
            calc_center_lat = (self.battlefield['min_lat'] + self.battlefield['max_lat']) / 2
            calc_center_lon = (self.battlefield['min_lon'] + self.battlefield['max_lon']) / 2
            print(f"[BTDemoAgent] 计算中心点: lat={calc_center_lat}, lon={calc_center_lon}")

        # --- 状态变量 ---
        self.initial_deployment_complete = False
        self.frame_count = 0
        self.own_units = []
        self.enemy_units = []
        self.enemy_missiles = []
        
        # --- 行为树上下文 (Blackboard) ---
        self.current_actions = []      # 本帧生成的指令列表
        self.commanded_units = set()   # 本帧已分配移动任务的单位名称
        
        # --- 战场辅助信息 ---
        self.center_lat = (self.battlefield['min_lat'] + self.battlefield['max_lat']) / 2
        self.center_lon = (self.battlefield['min_lon'] + self.battlefield['max_lon']) / 2
        self.defense_angle_offset = random.randint(0, 360)

        # --- 构建行为树 ---
        # 逻辑：
        # 1. 优先检查是否需要开局部署。
        # 2. 如果不需要部署，进入战斗循环 (Sequence)：
        #    a. 重置帧数据
        #    b. 处理导弹规避 (占用受威胁单位)
        #    c. 处理攻击 (占用剩余单位，但全员可开火)
        #    d. 处理剩余闲置单位 (巡逻)
        
        self.bt_root = Selector([
            # 分支 1: 开局部署
            Sequence([
                ConditionCheckInitialDeployment(),
                ActionExecuteDeployment()
            ]),

            # 分支 2: 常规战斗循环 (Main Loop)
            Sequence([
                ActionResetFrame(),          # 步骤1: 清理
                ActionEvadeMissiles(),       # 步骤2: 导弹规避（最高优先级）
                ActionProtectMannedVision(), # 步骤3: 无弹药无人机→保护有人机视野
                ActionMannedRetreat(),       # 步骤4: 发现敌机→有人机后撤
                ActionAttackLogic(),         # 步骤5: 开火逻辑
                ActionSearchFormation(),     # 步骤6: 无敌机→分散搜索推进
                ActionCenterPatrol(),        # 步骤7: 到达中心无敌机→盘旋
                ActionPatrolFormation()      # 步骤8: 兜底
            ])
        ])

    def update_decision(self, new_observation: Dict):
        """主入口函数"""
        self.observation = new_observation
        self.frame_count += 1
        
        # 1. 解析数据
        self._parse_observation(new_observation)
        
        if not self.own_units:
            return []

        # 2. 运行行为树
        self.bt_root.tick(self)
        
        # 3. 返回行为树生成的指令列表
        return self.current_actions

    # --- 辅助方法 (给节点调用) ---

    def add_action(self, cmd_dict, unit_name_occupy=None):
        """添加指令到列表，并可选地标记单位为'已占用'"""
        self.current_actions.append(cmd_dict)
        if unit_name_occupy:
            self.commanded_units.add(unit_name_occupy)

    def _parse_observation(self, obs):
        """解析观测数据"""
        assert obs.get('side') == self.side, f"side must be {self.side}, got {obs.get('side')}"

        self.own_units = obs.get('platform_list', [])
        self.enemy_units = []
        self.enemy_missiles = []
        for track in obs.get('track_list', []):
            if track.get('platform_entity_side') != self.side:
                if track.get('platform_entity_type') == '导弹':
                    self.enemy_missiles.append(track)
                else:
                    self.enemy_units.append(track)

    def try_fire_weapon(self, unit):
        """尝试扣除武器库存，成功返回True"""
        for weapon in unit.get('weapons', []):
            if weapon.get('quantity', 0) > 0:
                weapon['quantity'] -= 1
                return True
        return False

    def predict_threatened_units(self, missile):
        """预测威胁 (移植自原代码)"""
        threatened = []
        m_lon, m_lat = missile['longitude'], missile['latitude']
        m_heading = (math.degrees(missile.get('heading', 0)) + 360) % 360
        
        for unit in self.own_units:
            dist = YxGeoUtils.haversine_distance(unit['longitude'], unit['latitude'], m_lon, m_lat)
            bearing = YxGeoUtils.calculate_bearing(m_lon, m_lat, unit['longitude'], unit['latitude'])
            angle_diff = abs((bearing - m_heading + 180) % 360 - 180) # 归一化角度差
            
            if dist < 10000 and angle_diff < 15:
                threatened.append(unit)
        return threatened

    def calculate_evade_direction(self, unit, missile):
        """
        计算规避方向：垂直于导弹飞行方向（左或右）

        策略：
        - 获取导弹航向
        - 计算垂直于导弹航向的左右两个方向
        - 选择离战场边界更远的方向（避免飞出边界）
        """
        # 导弹航向（弧度转角度）
        missile_heading = math.degrees(missile.get('heading', 0)) % 360

        # 垂直于导弹航向的两个方向
        evade_left = (missile_heading - 90) % 360   # 导弹左侧
        evade_right = (missile_heading + 90) % 360  # 导弹右侧

        unit_lat = unit.get('latitude', self.center_lat)
        unit_lon = unit.get('longitude', self.center_lon)

        # 计算两个规避方向的目标点
        left_lon_off, left_lat_off = YxGeoUtils.km_to_lon_lat(self.center_lat, 5, evade_left)
        right_lon_off, right_lat_off = YxGeoUtils.km_to_lon_lat(self.center_lat, 5, evade_right)

        left_target_lat = unit_lat + left_lat_off
        left_target_lon = unit_lon + left_lon_off
        right_target_lat = unit_lat + right_lat_off
        right_target_lon = unit_lon + right_lon_off

        # 检查哪个方向更安全（离边界更远）
        left_safe = (self.battlefield['min_lat'] < left_target_lat < self.battlefield['max_lat'] and
                     self.battlefield['min_lon'] < left_target_lon < self.battlefield['max_lon'])
        right_safe = (self.battlefield['min_lat'] < right_target_lat < self.battlefield['max_lat'] and
                      self.battlefield['min_lon'] < right_target_lon < self.battlefield['max_lon'])

        if left_safe and not right_safe:
            return evade_left
        elif right_safe and not left_safe:
            return evade_right
        else:
            # 两边都安全或都不安全，选择离战场中心更近的方向
            left_dist_to_center = abs(left_target_lat - self.center_lat) + abs(left_target_lon - self.center_lon)
            right_dist_to_center = abs(right_target_lat - self.center_lat) + abs(right_target_lon - self.center_lon)

            return evade_left if left_dist_to_center < right_dist_to_center else evade_right