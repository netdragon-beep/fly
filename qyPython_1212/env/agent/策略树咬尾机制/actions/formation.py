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
    2. 有人机在后方35km处，保持安全距离
    3. 形成前后错层的搜索阵型，无人机先发现敌人
    4. 横向展开宽度基于雷达覆盖范围计算
    5. 有人机绝对不能超前，超前则原地盘旋等待

    阵型设计（基于雷达参数）：
    - 无人机雷达40km，方位±60°，有效宽度约40*sin(60°)*2=69km
    - 4架无人机横向展开，间距约15-20km，总宽度60-80km
    - 有人机在后方35km，利用60km雷达可覆盖无人机前方区域
    """

    # 基于官方参数计算的阵型参数
    UAV_RADAR_RANGE_KM = 40         # 无人机雷达距离 km
    MANNED_RADAR_RANGE_KM = 60      # 有人机雷达距离 km

    # 搜索阵型配置
    UAV_SPACING_KM = 18             # 无人机横向间距 km（覆盖雷达盲区）
    MANNED_BEHIND_KM = 35           # 有人机在无人机后方距离 km
    ADVANCE_SPEED_UAV = 300         # 无人机推进速度 m/s（中速推进，留有机动余量）
    ADVANCE_SPEED_MANNED = 200      # 有人机推进速度 m/s（低速跟进，绝不超车）
    LOITER_SPEED = 250              # 盘旋等待速度 m/s

    # 搜索终止条件：经过中心区域后继续搜索的距离
    PASS_CENTER_KM = 30             # 经过中心30km后停止搜索

    # 高度配置（在官方限制范围内）
    UAV_SEARCH_ALT = 3500           # 无人机搜索高度 m
    MANNED_SEARCH_ALT = 4500        # 有人机搜索高度 m（略高，便于俯视）

    # 超前判定阈值
    MIN_BEHIND_KM = 20              # 有人机至少在无人机后方20km

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
            # 计算无人机平均经度（用于超前检测）
            if uavs:
                uav_avg_lon = sum(u.get('longitude', 0) for u in uavs) / len(uavs)
            else:
                uav_avg_lon = avg_lon

            for unit in manned:
                manned_lon = unit.get('longitude', 0)

                # 检查有人机是否超前
                relative_pos_km = (manned_lon - uav_avg_lon) * enemy_dir * 111.0

                if relative_pos_km > -self.MIN_BEHIND_KM:
                    # 超前了！原地盘旋等待，不要继续前进
                    self._loiter_in_place(agent, unit)
                else:
                    # 正常跟随，但速度要慢
                    target_lat = center_lat
                    target_lon = manned_line_lon
                    target_pt = (target_lat, target_lon, self.MANNED_SEARCH_ALT)
                    agent.add_action(
                        decCmd.fly_to_point(unit['name'], target_pt, self.ADVANCE_SPEED_MANNED),
                        unit['name']
                    )

        return NodeStatus.SUCCESS

    def _loiter_in_place(self, agent, unit):
        """有人机原地盘旋等待无人机"""
        unit_name = unit.get('name', '')
        unit_lat = unit.get('latitude', agent.center_lat)
        unit_lon = unit.get('longitude', agent.center_lon)

        # 盘旋角度控制
        if not hasattr(agent, 'search_loiter_angle'):
            agent.search_loiter_angle = 0
        agent.search_loiter_angle = (agent.search_loiter_angle + 5) % 360

        # 计算盘旋位置（原地画圈，半径5km）
        radius_deg = 5.0 / 111.0
        angle_rad = math.radians(agent.search_loiter_angle)

        target_lat = unit_lat + radius_deg * math.sin(angle_rad)
        target_lon = unit_lon + radius_deg * math.cos(angle_rad)

        agent.add_action(
            decCmd.fly_to_point(
                unit_name,
                (target_lat, target_lon, self.MANNED_SEARCH_ALT),
                self.LOITER_SPEED
            ),
            unit_name
        )

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


class ActionCenterPriority(Action):
    """
    有人机中心优先节点 - 条件进攻策略

    核心战术：
    - 有人机是胜负关键（被击毁判负，中心停留时间决定平局胜负）
    - 有人机必须始终在无人机后方，绝不能超前
    - 如果有人机超前或没有视野，必须盘旋等待
    - 只有在确认安全后才进入战场
    - 残局时（敌机多数无弹药或逃跑）全速追击

    行为逻辑（按优先级）：
    1. 检查是否残局 → 残局则全速追击
    2. 检查是否超前 → 超前则盘旋等待
    3. 检查威胁等级 → DANGEROUS则保持后方
    4. 检查是否安全 → SAFE/CONDITIONAL则可以前进

    威胁评估等级：
    - SAFE: 战场无敌机 → 飞向中心
    - CONDITIONAL: 敌机都无弹药，或仅1架有弹药敌机 → 谨慎接近
    - DANGEROUS: 多架有弹药敌机 → 保持后方

    残局判定：
    - 敌方超过半数飞机没有导弹
    - 或者敌方正在逃跑（远离我方）
    """

    # 中心区域参数
    CENTER_RADIUS = 5000            # 中心区域半径 5km（官方定义）
    THREAT_DISTANCE = 40000         # 威胁评估距离 40km

    # 速度配置
    RETURN_SPEED = 400              # 返回中心速度 m/s（快速）
    CAUTIOUS_SPEED = 300            # 谨慎接近速度 m/s
    PATROL_SPEED = 250              # 盘旋速度 m/s（低速省油）
    LOITER_SPEED = 250              # 等待盘旋速度 m/s
    CHASE_SPEED = 500               # 残局追击速度 m/s（全速）

    # 盘旋参数
    PATROL_RADIUS_KM = 3            # 盘旋半径 3km
    PATROL_ALTITUDE = 4000          # 盘旋高度 m
    LOITER_RADIUS_KM = 5            # 等待盘旋半径 5km

    # 安全后方距离
    SAFE_BEHIND_KM = 35             # 保持在无人机后方35km
    MIN_BEHIND_KM = 20              # 最小后方距离20km（超前判定阈值）

    # 残局判定参数
    ENDGAME_UNARMED_RATIO = 0.5     # 敌方超过50%无弹药即为残局
    FLEEING_DISTANCE_THRESHOLD = 60000  # 敌机距离超过60km视为逃跑

    def tick(self, agent) -> str:
        # 获取未被命令的有人机
        manned_units = [u for u in agent.own_units
                        if u.get('type') == '有人机'
                        and u['name'] not in agent.commanded_units]

        if not manned_units:
            return NodeStatus.SUCCESS

        # 获取真实战场中心点
        battlefield = agent.battlefield
        if battlefield.get('min_lon') is not None:
            center_lon = (battlefield['min_lon'] + battlefield['max_lon']) / 2
            center_lat = (battlefield['min_lat'] + battlefield['max_lat']) / 2
        else:
            center_lon = agent.center_lon
            center_lat = agent.center_lat

        for unit in manned_units:
            # === 第零优先级：检查是否残局 ===
            endgame_status = self._check_endgame(agent, unit)
            if endgame_status == 'CHASE':
                # 残局！全速追击
                self._chase_enemy(agent, unit)
                continue
            elif endgame_status == 'OCCUPY':
                # 敌人在逃或无敌机，占领中心
                self._fly_to_center_fast(agent, unit, center_lon, center_lat)
                continue

            # === 第一优先级：检查是否超前 ===
            if self._is_ahead_of_uavs(agent, unit):
                # 超前了！必须盘旋等待无人机
                self._loiter_and_wait(agent, unit)
                continue

            # === 第二优先级：评估威胁 ===
            safety_status = self._evaluate_entry_safety(agent, unit)

            if safety_status == 'DANGEROUS':
                # 危险：保持在安全后方
                self._stay_behind(agent, unit)
            elif safety_status == 'SAFE':
                # 安全：飞向中心或盘旋
                dist_to_center = self._distance_to_center(unit, center_lon, center_lat)
                if dist_to_center < self.CENTER_RADIUS:
                    self._patrol_center(agent, unit, center_lon, center_lat)
                else:
                    self._fly_to_center(agent, unit, center_lon, center_lat)
            else:  # CONDITIONAL
                # 条件安全：谨慎接近中心
                self._approach_cautiously(agent, unit, center_lon, center_lat)

        return NodeStatus.SUCCESS

    def _check_endgame(self, agent, unit) -> str:
        """
        检查是否进入残局阶段

        返回:
        - 'CHASE': 残局，应该全速追击敌机
        - 'OCCUPY': 敌人在逃跑或无敌机，应该占领中心
        - 'NORMAL': 正常战斗，继续常规逻辑
        """
        if not agent.enemy_units:
            return 'OCCUPY'  # 无敌机，占领中心

        u_lon = unit.get('longitude', 0)
        u_lat = unit.get('latitude', 0)

        total_enemies = len(agent.enemy_units)
        enemies_without_ammo = 0
        enemies_fleeing = 0
        closest_armed_enemy = None
        closest_armed_dist = float('inf')

        for enemy in agent.enemy_units:
            e_lon = enemy.get('longitude', 0)
            e_lat = enemy.get('latitude', 0)
            if not e_lon or not e_lat:
                continue

            dist = YxGeoUtils.haversine_distance(u_lon, u_lat, e_lon, e_lat)
            has_ammo = self._enemy_has_ammo(enemy)

            if not has_ammo:
                enemies_without_ammo += 1

            # 检查敌机是否在逃跑（距离很远）
            if dist > self.FLEEING_DISTANCE_THRESHOLD:
                enemies_fleeing += 1

            # 记录最近的有弹药敌机
            if has_ammo and dist < closest_armed_dist:
                closest_armed_dist = dist
                closest_armed_enemy = enemy

        # 残局判定1：超过半数敌机无弹药
        unarmed_ratio = enemies_without_ammo / total_enemies if total_enemies > 0 else 0
        if unarmed_ratio >= self.ENDGAME_UNARMED_RATIO:
            # 如果还有有弹药的敌机很近，先不追
            if closest_armed_enemy and closest_armed_dist < self.THREAT_DISTANCE:
                return 'NORMAL'
            return 'CHASE'

        # 残局判定2：所有敌机都在逃跑
        if enemies_fleeing == total_enemies:
            return 'OCCUPY'  # 敌人全在跑，先占中心

        # 残局判定3：敌机全部无弹药
        if enemies_without_ammo == total_enemies:
            return 'CHASE'

        return 'NORMAL'

    def _chase_enemy(self, agent, unit):
        """残局：全速追击最近的敌机"""
        unit_name = unit.get('name', '')
        u_lon = unit.get('longitude', 0)
        u_lat = unit.get('latitude', 0)

        # 找最近的敌机
        closest_enemy = None
        min_dist = float('inf')

        for enemy in agent.enemy_units:
            e_lon = enemy.get('longitude', 0)
            e_lat = enemy.get('latitude', 0)
            if not e_lon or not e_lat:
                continue

            dist = YxGeoUtils.haversine_distance(u_lon, u_lat, e_lon, e_lat)
            if dist < min_dist:
                min_dist = dist
                closest_enemy = enemy

        if closest_enemy:
            e_lon = closest_enemy.get('longitude', 0)
            e_lat = closest_enemy.get('latitude', 0)
            e_alt = closest_enemy.get('altitude', 4000)

            agent.add_action(
                decCmd.fly_to_point(
                    unit_name,
                    (e_lat, e_lon, e_alt),
                    self.CHASE_SPEED  # 全速追击！
                ),
                unit_name
            )
        else:
            # 没有敌机，飞向中心
            agent.add_action(
                decCmd.fly_to_point(
                    unit_name,
                    (agent.center_lat, agent.center_lon, self.PATROL_ALTITUDE),
                    self.CHASE_SPEED
                ),
                unit_name
            )

    def _fly_to_center_fast(self, agent, unit, center_lon, center_lat):
        """快速飞向中心（残局占领）"""
        unit_name = unit.get('name', '')

        agent.add_action(
            decCmd.fly_to_point(
                unit_name,
                (center_lat, center_lon, self.PATROL_ALTITUDE),
                self.CHASE_SPEED  # 全速
            ),
            unit_name
        )

    def _is_ahead_of_uavs(self, agent, unit) -> bool:
        """
        检查有人机是否超前于无人机阵线

        判定条件：有人机经度超过无人机平均经度+安全距离
        """
        uavs = [u for u in agent.own_units if u.get('type') == '无人机']
        if not uavs:
            return False  # 没有无人机，不算超前

        # 计算无人机平均经度
        uav_avg_lon = sum(u.get('longitude', 0) for u in uavs) / len(uavs)

        # 有人机经度
        manned_lon = unit.get('longitude', 0)

        # 判断方向（红方向东，蓝方向西）
        enemy_dir = 1 if agent.side == 'red' else -1

        # 计算有人机相对于无人机的前后位置（正数=在前，负数=在后）
        relative_pos_km = (manned_lon - uav_avg_lon) * enemy_dir * 111.0

        # 如果有人机在无人机前方超过-MIN_BEHIND_KM（即不够后方），则判定为超前
        # relative_pos_km > -20 表示有人机没有保持20km以上的后方距离
        return relative_pos_km > -self.MIN_BEHIND_KM

    def _loiter_and_wait(self, agent, unit):
        """有人机盘旋等待，等无人机建立视野"""
        unit_name = unit.get('name', '')
        unit_lat = unit.get('latitude', agent.center_lat)
        unit_lon = unit.get('longitude', agent.center_lon)

        # 盘旋角度控制
        if not hasattr(agent, 'manned_loiter_angle'):
            agent.manned_loiter_angle = 0
        agent.manned_loiter_angle = (agent.manned_loiter_angle + 5) % 360

        # 计算盘旋位置（原地画圈）
        radius_deg = self.LOITER_RADIUS_KM / 111.0
        angle_rad = math.radians(agent.manned_loiter_angle)

        target_lat = unit_lat + radius_deg * math.sin(angle_rad)
        target_lon = unit_lon + radius_deg * math.cos(angle_rad)

        agent.add_action(
            decCmd.fly_to_point(
                unit_name,
                (target_lat, target_lon, self.PATROL_ALTITUDE),
                self.LOITER_SPEED
            ),
            unit_name
        )

        return NodeStatus.SUCCESS

    def _evaluate_entry_safety(self, agent, unit) -> str:
        """
        评估有人机入场安全性

        返回:
        - 'SAFE': 安全，可以入场（无敌机或敌机都无弹药）
        - 'CONDITIONAL': 条件安全（最多1架有弹药敌机威胁）
        - 'DANGEROUS': 危险（多架有弹药敌机威胁）
        """
        u_lon = unit.get('longitude', 0)
        u_lat = unit.get('latitude', 0)

        # 统计威胁范围内有弹药的敌机数量
        enemies_with_ammo = 0

        for enemy in agent.enemy_units:
            e_lon = enemy.get('longitude', 0)
            e_lat = enemy.get('latitude', 0)
            if not e_lon or not e_lat:
                continue

            dist = YxGeoUtils.haversine_distance(u_lon, u_lat, e_lon, e_lat)

            if dist < self.THREAT_DISTANCE:  # 40km内
                if self._enemy_has_ammo(enemy):
                    enemies_with_ammo += 1

        # 判断安全等级
        if enemies_with_ammo == 0:
            return 'SAFE'  # 无有弹药敌机
        elif enemies_with_ammo == 1:
            return 'CONDITIONAL'  # 只有1架有弹药，可以条件进攻
        else:
            return 'DANGEROUS'  # 多架有弹药，太危险

    def _enemy_has_ammo(self, enemy) -> bool:
        """检查敌机是否有弹药"""
        weapons = enemy.get('weapons', [])
        for w in weapons:
            if w.get('quantity', 0) > 0:
                return True
        # 如果没有武器信息，保守假设有弹药
        if not weapons:
            return True
        return False

    def _distance_to_center(self, unit, center_lon, center_lat) -> float:
        """计算到中心的距离"""
        u_lon = unit.get('longitude', 0)
        u_lat = unit.get('latitude', 0)
        return YxGeoUtils.haversine_distance(u_lon, u_lat, center_lon, center_lat)

    def _fly_to_center(self, agent, unit, center_lon, center_lat):
        """安全情况：快速飞向中心"""
        unit_name = unit.get('name', '')

        agent.add_action(
            decCmd.fly_to_point(
                unit_name,
                (center_lat, center_lon, self.PATROL_ALTITUDE),
                self.RETURN_SPEED
            ),
            unit_name
        )

    def _approach_cautiously(self, agent, unit, center_lon, center_lat):
        """条件安全：谨慎接近中心（低速，保持机动余量）"""
        unit_name = unit.get('name', '')

        agent.add_action(
            decCmd.fly_to_point(
                unit_name,
                (center_lat, center_lon, self.PATROL_ALTITUDE),
                self.CAUTIOUS_SPEED  # 较低速度，便于规避
            ),
            unit_name
        )

    def _stay_behind(self, agent, unit):
        """危险情况：保持在无人机后方35km"""
        unit_name = unit.get('name', '')

        # 找到无人机平均位置
        uavs = [u for u in agent.own_units if u.get('type') == '无人机']
        if uavs:
            uav_avg_lon = sum(u.get('longitude', 0) for u in uavs) / len(uavs)
            uav_avg_lat = sum(u.get('latitude', 0) for u in uavs) / len(uavs)

            # 保持在无人机后方35km
            enemy_dir = 1 if agent.side == 'red' else -1
            safe_lon = uav_avg_lon - OfficialParams.km_to_deg(self.SAFE_BEHIND_KM) * enemy_dir

            agent.add_action(
                decCmd.fly_to_point(
                    unit_name,
                    (uav_avg_lat, safe_lon, 4500),
                    280
                ),
                unit_name
            )
        else:
            # 无无人机，回到中心后方20km
            enemy_dir = 1 if agent.side == 'red' else -1
            safe_lon = agent.center_lon - OfficialParams.km_to_deg(20) * enemy_dir
            agent.add_action(
                decCmd.fly_to_point(
                    unit_name,
                    (agent.center_lat, safe_lon, 4500),
                    280
                ),
                unit_name
            )

    def _patrol_center(self, agent, unit, center_lon, center_lat):
        """中心内环盘旋"""
        unit_name = unit.get('name', '')

        # 盘旋角度控制
        if not hasattr(agent, 'manned_center_angle'):
            agent.manned_center_angle = 0
        agent.manned_center_angle = (agent.manned_center_angle + 3) % 360

        # 计算盘旋位置
        radius_deg = OfficialParams.km_to_deg(self.PATROL_RADIUS_KM)
        angle_rad = math.radians(agent.manned_center_angle)

        target_lat = center_lat + radius_deg * math.sin(angle_rad)
        target_lon = center_lon + radius_deg * math.cos(angle_rad)

        agent.add_action(
            decCmd.fly_to_point(
                unit_name,
                (target_lat, target_lon, self.PATROL_ALTITUDE),
                self.PATROL_SPEED
            ),
            unit_name
        )


class ActionCenterPatrol(Action):
    """
    无人机外围警戒节点

    注意：有人机的中心控制由 ActionCenterPriority 节点处理

    战术设计：
    1. 无人机在中心外围（18km）形成警戒圈
    2. 即使有敌机也可以执行（空闲无人机警戒）
    3. 为有人机提供外层保护和预警
    """

    # 警戒参数
    UAV_PATROL_RADIUS_KM = 18       # 无人机警戒半径 km（比中心5km远，比攻击范围近）

    # 速度配置
    UAV_PATROL_SPEED = 320          # 无人机巡逻速度 m/s

    # 高度配置
    UAV_PATROL_ALT = 3500           # 无人机巡逻高度 m

    def tick(self, agent) -> str:
        # 只处理未被占用的无人机
        uavs = [u for u in agent.own_units if u.get('type') == '无人机'
                and u['name'] not in agent.commanded_units]

        if not uavs:
            return NodeStatus.SUCCESS

        # 获取真实战场中心点
        battlefield = agent.battlefield
        if battlefield.get('min_lon') is not None:
            real_center_lon = (battlefield['min_lon'] + battlefield['max_lon']) / 2
            real_center_lat = (battlefield['min_lat'] + battlefield['max_lat']) / 2
        else:
            real_center_lon = agent.center_lon
            real_center_lat = agent.center_lat

        # 盘旋角度控制
        if not hasattr(agent, 'uav_patrol_angle'):
            agent.uav_patrol_angle = 0
        agent.uav_patrol_angle = (agent.uav_patrol_angle + 2) % 360

        # === 无人机外围警戒 ===
        outer_radius_deg = OfficialParams.km_to_deg(self.UAV_PATROL_RADIUS_KM)

        for i, uav in enumerate(uavs):
            # 均匀分布在圆周上
            angle = (agent.uav_patrol_angle + i * (360 / len(uavs))) % 360
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
