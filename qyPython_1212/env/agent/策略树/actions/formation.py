"""
阵型和战术动作节点

基于官方赛题参数设计（来自初赛赛题任务想定设计.pdf）：

=== 官方参数 ===
飞机参数：
- 有人机：速度180-500m/s，高度2000-7000m，雷达60km，方位±60°，俯仰±50°
- 无人机：速度120-360m/s，高度2000-7000m，雷达40km，方位±60°，俯仰±40°

战场参数：
- 任务区域：200km × 200km
- 中心区域：半径5km的圆
- 初始部署高度：4000m
- 想定时长：15分钟

胜负规则：
- 有人机被击毁则立即判负
- 导弹耗尽/时间结束：无人机数量多者胜
- 无人机数量相同：有人机在中心区域停留时间长者胜

=== 战术设计原则 ===
1. 有人机存活是最高优先级（被击毁直接判负）
2. 无人机前置保护有人机
3. 利用雷达探测范围形成信息优势
4. 争夺中心区域控制权

包含：
- ActionSearchFormation: 分散搜索阵型
- ActionMannedRetreat: 有人机后撤保护
- ActionProtectMannedVision: 无人机保护有人机
- ActionCenterPatrol: 中心区域控制
- ActionPatrolFormation: 防御巡逻阵型
"""

import math
from ..bt_framework import Action, NodeStatus
from utilities.yxScriptTreeFunc import YxScriptTreeFunc as decCmd
from utilities.yxGeoUtils import YxGeoUtils


# ========== 官方参数常量 ==========
class OfficialParams:
    """官方赛题参数（来自PDF）"""

    # 有人机参数
    MANNED_MIN_SPEED = 180          # m/s
    MANNED_MAX_SPEED = 500          # m/s
    MANNED_MIN_ALT = 2000           # m
    MANNED_MAX_ALT = 7000           # m
    MANNED_RADAR_RANGE = 60000      # m (60km)
    MANNED_RADAR_AZIMUTH = 60       # ±60°

    # 无人机参数
    UAV_MIN_SPEED = 120             # m/s
    UAV_MAX_SPEED = 360             # m/s
    UAV_MIN_ALT = 2000              # m
    UAV_MAX_ALT = 7000              # m
    UAV_RADAR_RANGE = 40000         # m (40km)
    UAV_RADAR_AZIMUTH = 60          # ±60°

    # 导弹参数
    MISSILE_SPEED = 1200            # m/s
    MISSILE_MAX_RANGE = 72000       # m (72km)
    MISSILE_MAX_TIME = 60           # s

    # 战场参数
    BATTLEFIELD_SIZE = 200000       # m (200km)
    CENTER_RADIUS = 5000            # m (5km)
    INITIAL_ALTITUDE = 4000         # m

    # 经纬度转换（约111km/度）
    KM_PER_DEGREE = 111.0
    DEG_PER_KM = 1.0 / 111.0

    @classmethod
    def km_to_deg(cls, km):
        """公里转经纬度度数"""
        return km * cls.DEG_PER_KM

    @classmethod
    def m_to_deg(cls, m):
        """米转经纬度度数"""
        return m / 1000.0 * cls.DEG_PER_KM


class ActionSearchFormation(Action):
    """
    分散搜索阵型

    战术思路：
    1. 无人机在前方展开搜索线（利用40km雷达）
    2. 有人机在后方10-15km处（利用60km雷达可以覆盖更远）
    3. 形成前后错层的搜索阵型，无人机先发现敌人
    4. 横向展开宽度基于雷达覆盖范围计算

    阵型设计（基于雷达参数）：
    - 无人机雷达40km，方位±60°，有效宽度约40*sin(60°)*2=69km
    - 4架无人机横向展开，间距约15-20km，总宽度60-80km
    - 有人机在后方，雷达60km可覆盖无人机前方区域
    """

    # 基于官方参数计算的阵型参数
    UAV_RADAR_RANGE_KM = 40         # 无人机雷达距离 km
    MANNED_RADAR_RANGE_KM = 60      # 有人机雷达距离 km

    # 搜索阵型配置
    UAV_SPACING_KM = 18             # 无人机横向间距 km（覆盖雷达盲区）
    MANNED_BEHIND_KM = 15           # 有人机在无人机后方距离 km
    ADVANCE_SPEED_UAV = 300         # 无人机推进速度 m/s（中速推进，留有机动余量）
    ADVANCE_SPEED_MANNED = 280      # 有人机推进速度 m/s（略慢于无人机）

    # 搜索终止条件：经过中心区域后继续搜索的距离
    PASS_CENTER_KM = 30             # 经过中心30km后停止搜索

    # 高度配置（在官方限制范围内）
    UAV_SEARCH_ALT = 3500           # 无人机搜索高度 m
    MANNED_SEARCH_ALT = 4500        # 有人机搜索高度 m（略高，便于俯视）

    def tick(self, agent) -> str:
        # 如果已发现敌机，不执行搜索阵型（交给攻击逻辑）
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

        # 确定敌方方向（红方向东攻击，蓝方向西攻击）
        enemy_dir = 1 if agent.side == 'red' else -1

        # 计算当前阵线位置
        all_units = manned + uavs
        avg_lon = sum(u.get('longitude', 0) for u in all_units) / len(all_units)

        # 检查是否已经过中心区域足够远
        passed_center_deg = (avg_lon - center_lon) * enemy_dir
        passed_center_km = passed_center_deg * OfficialParams.KM_PER_DEGREE

        if passed_center_km > self.PASS_CENTER_KM:
            return NodeStatus.SUCCESS  # 交给ActionCenterPatrol处理

        # === 计算搜索阵型位置 ===
        # 无人机前置搜索线
        uav_line_lon = avg_lon + OfficialParams.km_to_deg(5) * enemy_dir

        # 有人机后置支援线
        manned_line_lon = uav_line_lon - OfficialParams.km_to_deg(self.MANNED_BEHIND_KM) * enemy_dir

        # === 部署无人机搜索线 ===
        if uavs:
            num_uavs = len(uavs)
            # 计算横向展开宽度
            total_width_km = (num_uavs - 1) * self.UAV_SPACING_KM
            total_width_deg = OfficialParams.km_to_deg(total_width_km)
            start_lat = center_lat - total_width_deg / 2

            # 按纬度排序，保持阵型稳定
            uavs_sorted = sorted(uavs, key=lambda u: u.get('latitude', 0))

            for i, uav in enumerate(uavs_sorted):
                target_lat = start_lat + i * OfficialParams.km_to_deg(self.UAV_SPACING_KM)
                target_lon = uav_line_lon
                target_pt = (target_lat, target_lon, self.UAV_SEARCH_ALT)
                agent.add_action(
                    decCmd.fly_to_point(uav['name'], target_pt, self.ADVANCE_SPEED_UAV),
                    uav['name']
                )

        # === 部署有人机支援位置 ===
        if manned:
            for unit in manned:
                # 有人机在编队中央后方
                target_lat = center_lat
                target_lon = manned_line_lon
                target_pt = (target_lat, target_lon, self.MANNED_SEARCH_ALT)
                agent.add_action(
                    decCmd.fly_to_point(unit['name'], target_pt, self.ADVANCE_SPEED_MANNED),
                    unit['name']
                )

        return NodeStatus.SUCCESS

class ActionProtectMannedVision(Action):
    """
    无人机保护有人机

    战术设计：
    1. 无弹药的无人机不再执行攻击任务
    2. 返回有人机周围形成保护阵型
    3. 覆盖有人机雷达盲区（后方和两侧）
    4. 必要时可以为有人机挡弹

    保护阵型：
    - 雷达方位角±60°，后方120°是盲区
    - 无人机分布在有人机后方半球
    - 形成环形保护圈
    """

    # 保护参数
    PROTECT_RADIUS_KM = 8           # 保护圈半径 km
    PROTECT_SPEED = 300             # 巡逻速度 m/s
    PROTECT_ALTITUDE = 4000         # 保护高度 m（与有人机同高度便于协同）

    # 盲区覆盖角度
    BLIND_ZONE_START = 120          # 盲区起始角度（相对于敌方方向）
    BLIND_ZONE_END = 240            # 盲区结束角度

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
            # 无有人机（已被击毁），在中心区域待命
            protect_center_lat = agent.center_lat
            protect_center_lon = agent.center_lon

        # 盘旋角度控制
        if not hasattr(agent, 'protect_angle'):
            agent.protect_angle = 0
        agent.protect_angle = (agent.protect_angle + 4) % 360

        enemy_dir = 1 if agent.side == 'red' else -1
        num_protect = len(uavs_no_ammo)

        # 计算敌方方向的绝对角度
        # 红方敌人在东（90°），蓝方敌人在西（270°）
        enemy_angle = 90 if enemy_dir > 0 else 270

        for i, uav in enumerate(uavs_no_ammo):
            # 在有人机后方半球分布（覆盖盲区）
            # 盲区是相对于敌方方向的120°-240°
            if num_protect > 1:
                # 在盲区范围内均匀分布
                spread_range = self.BLIND_ZONE_END - self.BLIND_ZONE_START
                unit_offset = self.BLIND_ZONE_START + (i / (num_protect - 1)) * spread_range
            else:
                unit_offset = 180  # 单个无人机直接在后方

            # 转换为绝对角度
            abs_angle = (enemy_angle + unit_offset + agent.protect_angle * 0.2) % 360
            angle_rad = math.radians(abs_angle)

            # 计算保护位置
            protect_radius_deg = OfficialParams.km_to_deg(self.PROTECT_RADIUS_KM)
            target_lat = protect_center_lat + protect_radius_deg * math.sin(angle_rad)
            target_lon = protect_center_lon + protect_radius_deg * math.cos(angle_rad)

            target_pt = (target_lat, target_lon, self.PROTECT_ALTITUDE)
            agent.add_action(
                decCmd.fly_to_point(uav['name'], target_pt, self.PROTECT_SPEED),
                uav['name']
            )

        return NodeStatus.SUCCESS


class ActionCenterPatrol(Action):
    """
    中心区域控制

    胜负规则：无人机数量相同时，有人机在中心区域停留时间长者胜

    战术设计：
    1. 搜索完毕无敌机时，进入中心区域控制模式
    2. 有人机在中心区域（半径5km内）盘旋积累时间
    3. 无人机在外围提供保护和预警
    4. 形成内外双层防御圈

    中心区域定义（官方）：
    - 同一水平面内距离任务区中心点5km之内的圆
    """

    # 中心区域参数（官方定义）
    CENTER_RADIUS_KM = 5            # 中心区域半径 km

    # 巡逻阵型参数
    MANNED_PATROL_RADIUS_KM = 3     # 有人机巡逻半径 km（在5km中心区域内）
    UAV_PATROL_RADIUS_KM = 15       # 无人机巡逻半径 km（外围警戒）

    # 速度配置
    MANNED_PATROL_SPEED = 250       # 有人机巡逻速度 m/s（低速节省，便于停留）
    UAV_PATROL_SPEED = 320          # 无人机巡逻速度 m/s

    # 高度配置
    MANNED_PATROL_ALT = 4000        # 有人机巡逻高度 m
    UAV_PATROL_ALT = 3500           # 无人机巡逻高度 m

    # 触发条件
    PASS_CENTER_KM = 30             # 经过中心30km后开始盘旋

    def tick(self, agent) -> str:
        # 如果有敌机，不执行盘旋（优先攻击/防御）
        if agent.enemy_units:
            return NodeStatus.SUCCESS

        manned = [u for u in agent.own_units if u.get('type') == '有人机'
                  and u['name'] not in agent.commanded_units]
        uavs = [u for u in agent.own_units if u.get('type') == '无人机'
                and u['name'] not in agent.commanded_units]

        if not manned and not uavs:
            return NodeStatus.SUCCESS

        # 获取真实战场中心点
        battlefield = agent.battlefield
        if battlefield.get('min_lon') is not None:
            real_center_lon = (battlefield['min_lon'] + battlefield['max_lon']) / 2
            real_center_lat = (battlefield['min_lat'] + battlefield['max_lat']) / 2
        else:
            real_center_lon = agent.center_lon
            real_center_lat = agent.center_lat

        # 检查是否已经过中心区域足够远
        all_units = manned + uavs
        avg_lon = sum(u.get('longitude', 0) for u in all_units) / len(all_units)
        enemy_dir = 1 if agent.side == 'red' else -1

        passed_center_deg = (avg_lon - real_center_lon) * enemy_dir
        passed_center_km = passed_center_deg * OfficialParams.KM_PER_DEGREE

        if passed_center_km < self.PASS_CENTER_KM:
            return NodeStatus.SUCCESS  # 还没搜索完毕

        # === 进入中心控制模式 ===
        # 调试输出
        # if not hasattr(agent, '_center_patrol_logged'):
        #     agent._center_patrol_logged = True
        #     print(f"[CenterPatrol] 进入中心控制模式，中心点: ({real_center_lat:.4f}, {real_center_lon:.4f})")

        # 盘旋角度控制
        if not hasattr(agent, 'center_patrol_angle'):
            agent.center_patrol_angle = 0
        agent.center_patrol_angle = (agent.center_patrol_angle + 3) % 360

        # === 有人机内环巡逻（在中心5km区域内） ===
        if manned:
            inner_radius_deg = OfficialParams.km_to_deg(self.MANNED_PATROL_RADIUS_KM)

            for i, unit in enumerate(manned):
                angle = (agent.center_patrol_angle + i * (360 / len(manned))) % 360
                angle_rad = math.radians(angle)

                target_lat = real_center_lat + inner_radius_deg * math.sin(angle_rad)
                target_lon = real_center_lon + inner_radius_deg * math.cos(angle_rad)
                target_pt = (target_lat, target_lon, self.MANNED_PATROL_ALT)

                agent.add_action(
                    decCmd.fly_to_point(unit['name'], target_pt, self.MANNED_PATROL_SPEED),
                    unit['name']
                )

        # === 无人机外环警戒 ===
        if uavs:
            outer_radius_deg = OfficialParams.km_to_deg(self.UAV_PATROL_RADIUS_KM)

            for i, uav in enumerate(uavs):
                angle = (agent.center_patrol_angle + i * (360 / len(uavs))) % 360
                angle_rad = math.radians(angle)

                target_lat = real_center_lat + outer_radius_deg * math.sin(angle_rad)
                target_lon = real_center_lon + outer_radius_deg * math.cos(angle_rad)
                target_pt = (target_lat, target_lon, self.UAV_PATROL_ALT)

                agent.add_action(
                    decCmd.fly_to_point(uav['name'], target_pt, self.UAV_PATROL_SPEED),
                    uav['name']
                )

        return NodeStatus.SUCCESS


class ActionPatrolFormation(Action):
    """
    防御巡逻阵型（兜底策略）

    当其他策略都不适用时的默认行为：
    - 有人机在内环，靠近中心
    - 无人机在外环，提供警戒

    这是最保守的阵型，确保有人机安全
    """

    # 阵型参数
    MANNED_RADIUS_KM = 10           # 有人机巡逻半径 km
    UAV_RADIUS_KM = 25              # 无人机巡逻半径 km

    # 速度配置
    PATROL_SPEED_MANNED = 280       # 有人机巡逻速度 m/s
    PATROL_SPEED_UAV = 320          # 无人机巡逻速度 m/s

    # 高度配置
    MANNED_ALT = 4500               # 有人机高度 m
    UAV_ALT = 3500                  # 无人机高度 m

    def tick(self, agent) -> str:
        # 仅控制剩下的单位
        available_units = [u for u in agent.own_units if u['name'] not in agent.commanded_units]
        if not available_units:
            return NodeStatus.SUCCESS

        manned = [u for u in available_units if u.get('type') == '有人机']
        uavs = [u for u in available_units if u.get('type') == '无人机']

        center_lat = agent.center_lat
        center_lon = agent.center_lon

        # 动态旋转角度
        if not hasattr(agent, 'defense_angle_offset'):
            agent.defense_angle_offset = 0
        agent.defense_angle_offset = (agent.defense_angle_offset + 2) % 360

        # 有人机内环
        if manned:
            inner_radius_deg = OfficialParams.km_to_deg(self.MANNED_RADIUS_KM)
            for i, unit in enumerate(manned):
                angle = (agent.defense_angle_offset + i * (360 / len(manned))) % 360
                angle_rad = math.radians(angle)

                target_lat = center_lat + inner_radius_deg * math.sin(angle_rad)
                target_lon = center_lon + inner_radius_deg * math.cos(angle_rad)
                target_pt = (target_lat, target_lon, self.MANNED_ALT)

                agent.add_action(
                    decCmd.fly_to_point(unit['name'], target_pt, self.PATROL_SPEED_MANNED),
                    unit['name']
                )

        # 无人机外环
        if uavs:
            outer_radius_deg = OfficialParams.km_to_deg(self.UAV_RADIUS_KM)
            for i, unit in enumerate(uavs):
                angle = (agent.defense_angle_offset + i * (360 / len(uavs))) % 360
                angle_rad = math.radians(angle)

                target_lat = center_lat + outer_radius_deg * math.sin(angle_rad)
                target_lon = center_lon + outer_radius_deg * math.cos(angle_rad)
                target_pt = (target_lat, target_lon, self.UAV_ALT)

                agent.add_action(
                    decCmd.fly_to_point(unit['name'], target_pt, self.PATROL_SPEED_UAV),
                    unit['name']
                )

        return NodeStatus.SUCCESS
