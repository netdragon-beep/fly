"""
导弹规避动作节点

基于官方赛题参数设计的规避策略：
- 有人机：最大速度500m/s，高度范围2000-7000m
- 无人机：最大速度360m/s，高度范围2000-7000m
- 导弹：速度1200m/s，最大飞行时间60s，最大飞行距离72km

规避策略采用Cranking机动 + 高度变化 + 速度调整
"""

import math
from ..bt_framework import Action, NodeStatus
from utilities.yxScriptTreeFunc import YxScriptTreeFunc as decCmd
from utilities.yxGeoUtils import YxGeoUtils


class ActionEvadeMissiles(Action):
    """
    多导弹协同规避 - 基于官方参数的Cranking机动策略

    === 官方参数（来自初赛赛题任务想定设计.pdf）===

    飞机参数：
    - 有人机：速度180-500m/s，高度2000-7000m，最大线性加速度20m/s²，最大径向加速度15m/s²
    - 无人机：速度120-360m/s，高度2000-7000m，最大线性加速度20m/s²，最大径向加速度15m/s²

    导弹参数：
    - 平均飞行速度：1200 m/s
    - 最长飞行时间：60s
    - 最大飞行距离：72km
    - 杀伤半径：10m
    - 导引头引爆距离：10m

    战场参数：
    - 任务区域：200km × 200km
    - 雷达探测距离：有人机60km，无人机40km

    规避策略原理：
    1. 威胁评估：基于导弹最大飞行距离72km和飞行时间60s计算威胁区
    2. Cranking机动：斜向后方飞行，迫使导弹持续转向消耗能量
    3. 高度规避：利用飞机高度限制进行垂直机动
    4. 速度优化：根据威胁等级调整到最大速度
    5. 边界感知：确保规避方向不会飞出200km×200km战场
    """

    # ========== 官方飞机参数 ==========
    # 有人机
    MANNED_MIN_SPEED = 180          # 有人机最小速度 m/s
    MANNED_MAX_SPEED = 500          # 有人机最大速度 m/s
    MANNED_MIN_ALT = 2000           # 有人机最低高度 m
    MANNED_MAX_ALT = 7000           # 有人机最高高度 m
    MANNED_MAX_LINEAR_ACC = 20      # 有人机最大线性加速度 m/s²
    MANNED_MAX_RADIAL_ACC = 15      # 有人机最大径向加速度 m/s²

    # 无人机
    UAV_MIN_SPEED = 120             # 无人机最小速度 m/s
    UAV_MAX_SPEED = 360             # 无人机最大速度 m/s
    UAV_MIN_ALT = 2000              # 无人机最低高度 m
    UAV_MAX_ALT = 7000              # 无人机最高高度 m
    UAV_MAX_LINEAR_ACC = 20         # 无人机最大线性加速度 m/s²
    UAV_MAX_RADIAL_ACC = 15         # 无人机最大径向加速度 m/s²

    # ========== 官方导弹参数 ==========
    MISSILE_SPEED = 1200            # 导弹平均速度 m/s
    MISSILE_MAX_FLIGHT_TIME = 60    # 导弹最长飞行时间 s
    MISSILE_MAX_RANGE = 72000       # 导弹最大飞行距离 m (72km)
    MISSILE_KILL_RADIUS = 10        # 导弹杀伤半径 m
    MISSILE_SEEKER_RANGE = 10       # 导引头引爆距离 m

    # ========== 官方雷达参数 ==========
    MANNED_RADAR_RANGE = 60000      # 有人机雷达探测距离 m (60km)
    UAV_RADAR_RANGE = 40000         # 无人机雷达探测距离 m (40km)
    RADAR_AZIMUTH_RANGE = 60        # 雷达方位范围 ±60°
    MANNED_RADAR_PITCH_RANGE = 50   # 有人机雷达俯仰范围 ±50°
    UAV_RADAR_PITCH_RANGE = 40      # 无人机雷达俯仰范围 ±40°

    # ========== 官方战场参数 ==========
    BATTLEFIELD_SIZE = 200000       # 任务区域大小 m (200km)
    CENTER_RADIUS = 5000            # 中心区域半径 m (5km)
    INITIAL_ALTITUDE = 4000         # 初始部署高度 m
    MISSION_DURATION = 900          # 想定时长 s (15分钟)

    # ========== 威胁判断参数（基于官方导弹参数计算）==========
    # 导弹最大射程72km，但实际有效拦截距离需考虑：
    # - 导弹飞行时间60s，速度1200m/s
    # - 飞机最大逃逸速度（有人机500m/s）
    # - 导弹需要追踪转向消耗能量

    # 威胁半径 = 导弹有效攻击距离（考虑飞机规避能力）
    # 设定为导弹最大射程的一半左右，预留规避时间
    MISSILE_THREAT_RADIUS = 50000       # 威胁半径50km（开始关注）
    MISSILE_DANGER_RADIUS = 35000       # 危险半径35km（积极规避）
    MISSILE_CRITICAL_RADIUS = 25000     # 紧急半径25km（全力规避）
    MISSILE_LETHAL_RADIUS = 12000       # 致命半径12km（极限规避）

    # 导弹威胁角度：导弹航向与目标夹角
    # 雷达方位范围±60°，所以导弹也应该在这个范围内才有威胁
    MISSILE_THREAT_ANGLE = 75           # 导弹威胁角度（略大于雷达方位范围）

    # ========== 规避行为参数 ==========
    # 规避速度：根据威胁等级使用不同速度
    # 有人机规避速度
    MANNED_EVADE_SPEED_NORMAL = 400     # 正常规避速度 (80%最大速度)
    MANNED_EVADE_SPEED_DANGER = 450     # 危险规避速度 (90%最大速度)
    MANNED_EVADE_SPEED_CRITICAL = 500   # 紧急规避速度 (最大速度)

    # 无人机规避速度
    UAV_EVADE_SPEED_NORMAL = 350        # 正常规避速度 (83%最大速度)
    UAV_EVADE_SPEED_DANGER = 355        # 危险规避速度 (92%最大速度)
    UAV_EVADE_SPEED_CRITICAL = 360      # 紧急规避速度 (最大速度)

    # 规避距离：规避目标点距当前位置的距离
    # 基于导弹飞行时间和飞机速度计算
    # 导弹飞行10km约需8.3s，飞机需要足够距离完成转向
    EVADE_DISTANCE_NORMAL = 15          # 正常规避距离 km
    EVADE_DISTANCE_DANGER = 20          # 危险规避距离 km
    EVADE_DISTANCE_CRITICAL = 25        # 紧急规避距离 km

    # ========== Cranking机动参数 ==========
    # Cranking：向导弹反方向的斜后方飞行
    # 角度越大越接近直接逃跑，角度越小越接近横向机动
    # 最优Cranking角度约120-150°，能有效消耗导弹能量
    CRANK_ANGLE_NORMAL = 120            # 正常Crank角度（向后斜飞120°）
    CRANK_ANGLE_DANGER = 135            # 危险Crank角度（更向后135°）
    CRANK_ANGLE_CRITICAL = 150          # 紧急Crank角度（几乎直接逃跑150°）
    CRANK_ANGLE_LETHAL = 170            # 致命Crank角度（直接逃跑）

    # ========== 高度规避参数 ==========
    # 利用高度变化增加导弹追踪难度
    # 高度范围：2000-7000m（官方限制）
    ALTITUDE_CHANGE_NORMAL = 500        # 正常高度变化 m
    ALTITUDE_CHANGE_DANGER = 800        # 危险高度变化 m
    ALTITUDE_CHANGE_CRITICAL = 1200     # 紧急高度变化 m

    # 高度限制（官方参数）
    MIN_ALTITUDE = 2000                 # 最低飞行高度 m
    MAX_ALTITUDE = 7000                 # 最高飞行高度 m

    # ========== 边界安全参数 ==========
    # 战场边界安全边距
    # 无人机/导弹出界立即销毁，有人机出界30秒销毁
    # 设置10km安全边距确保规避时不会出界
    BOUNDARY_MARGIN_KM = 10             # 边界安全边距 km
    BOUNDARY_MARGIN = 0.09              # 边界安全边距（经纬度，约10km）

    # ========== 调试参数 ==========
    DEBUG_ENABLED = True               # 调试开关
    DEBUG_INTERVAL = 20                 # 调试输出间隔帧数

    def tick(self, agent) -> str:
        """
        执行导弹规避逻辑

        优先级：
        1. 检测所有来袭导弹
        2. 评估每个己方单位的威胁等级
        3. 计算最优规避方向（Cranking机动）
        4. 执行规避机动
        """
        # 调试输出
        if self.DEBUG_ENABLED:
            if not hasattr(agent, '_missile_debug_frame'):
                agent._missile_debug_frame = 0
            if agent.frame_count - agent._missile_debug_frame >= self.DEBUG_INTERVAL:
                agent._missile_debug_frame = agent.frame_count
                print(f"[导弹规避] Frame {agent.frame_count}: 检测到 {len(agent.enemy_missiles)} 枚敌方导弹")

        # 无导弹则返回成功
        if not agent.enemy_missiles:
            return NodeStatus.SUCCESS

        # 收集所有导弹信息
        all_missiles = self._collect_missile_info(agent)
        if not all_missiles:
            return NodeStatus.SUCCESS

        # 对每个己方单位进行威胁评估和规避
        evade_count = 0
        for unit in agent.own_units:
            if self._process_unit_evasion(agent, unit, all_missiles):
                evade_count += 1

        if self.DEBUG_ENABLED and evade_count > 0:
            print(f"[导弹规避] Frame {agent.frame_count}: {evade_count} 个单位执行规避机动")

        return NodeStatus.SUCCESS

    def _collect_missile_info(self, agent):
        """收集所有导弹信息"""
        missiles = []
        for missile in agent.enemy_missiles:
            m_lon = missile.get('longitude', 0)
            m_lat = missile.get('latitude', 0)
            m_heading = math.degrees(missile.get('heading', 0)) % 360
            m_speed = missile.get('speed', self.MISSILE_SPEED)
            m_alt = missile.get('altitude', self.INITIAL_ALTITUDE)

            if m_lon and m_lat:
                missiles.append({
                    'lon': m_lon,
                    'lat': m_lat,
                    'heading': m_heading,
                    'speed': m_speed,
                    'altitude': m_alt,
                    'id': missile.get('target_id') or missile.get('name') or id(missile)
                })
        return missiles

    def _process_unit_evasion(self, agent, unit, all_missiles):
        """
        处理单个单位的规避逻辑

        返回: True如果执行了规避，False否则
        """
        unit_name = unit['name']
        u_lon = unit.get('longitude', 0)
        u_lat = unit.get('latitude', 0)
        u_alt = unit.get('altitude', self.INITIAL_ALTITUDE)
        is_manned = unit.get('type') == '有人机'

        if not u_lon or not u_lat:
            return False

        # 分析威胁导弹
        threat_info = self._analyze_threats(u_lon, u_lat, all_missiles)

        if not threat_info['threats']:
            return False

        # 确定威胁等级和规避参数
        evade_params = self._get_evade_params(threat_info, is_manned)

        # 计算Cranking规避方向
        evade_direction = self._calculate_crank_direction(
            u_lon, u_lat,
            threat_info['primary_threat'],
            evade_params['crank_angle'],
            agent
        )

        # 计算规避目标点
        evade_lon, evade_lat = self._calculate_evade_point(
            u_lon, u_lat, u_lat,
            evade_direction,
            evade_params['distance'],
            agent
        )

        # 计算规避高度
        evade_alt = self._calculate_evade_altitude(
            u_alt,
            threat_info['min_dist'],
            evade_params['alt_change'],
            is_manned,
            unit_name
        )

        # 执行规避
        agent.add_action(
            decCmd.fly_to_point(unit_name, (evade_lat, evade_lon, evade_alt), evade_params['speed']),
            unit_name
        )

        # 调试输出
        if self.DEBUG_ENABLED:
            level = evade_params['level']
            print(f"[规避执行] [{level}] {unit_name}: "
                  f"距离={threat_info['min_dist']:.0f}m, "
                  f"Crank={evade_params['crank_angle']}°, "
                  f"速度={evade_params['speed']}m/s, "
                  f"目标=({evade_lat:.4f}, {evade_lon:.4f}, {evade_alt:.0f}m)")

        return True

    def _analyze_threats(self, u_lon, u_lat, all_missiles):
        """
        分析导弹威胁

        返回: {
            'threats': 威胁导弹列表,
            'primary_threat': 最近的威胁导弹,
            'min_dist': 最近导弹距离
        }
        """
        threats = []
        min_dist = float('inf')
        primary_threat = None

        for missile in all_missiles:
            dist = YxGeoUtils.haversine_distance(u_lon, u_lat, missile['lon'], missile['lat'])

            # 超出威胁半径则忽略
            if dist > self.MISSILE_THREAT_RADIUS:
                continue

            # 计算导弹航向与目标的夹角
            bearing_to_unit = YxGeoUtils.calculate_bearing(
                missile['lon'], missile['lat'], u_lon, u_lat
            )
            angle_diff = abs((bearing_to_unit - missile['heading'] + 180) % 360 - 180)

            # 只有在威胁角度内的导弹才构成威胁
            if angle_diff < self.MISSILE_THREAT_ANGLE:
                threat = {
                    **missile,
                    'dist': dist,
                    'bearing_to_unit': bearing_to_unit,
                    'angle_diff': angle_diff
                }
                threats.append(threat)

                if dist < min_dist:
                    min_dist = dist
                    primary_threat = threat

        return {
            'threats': threats,
            'primary_threat': primary_threat,
            'min_dist': min_dist if threats else float('inf')
        }

    def _get_evade_params(self, threat_info, is_manned):
        """
        根据威胁等级获取规避参数

        威胁等级：
        - LETHAL: < 8km (致命)
        - CRITICAL: < 15km (紧急)
        - DANGER: < 25km (危险)
        - NORMAL: < 40km (正常)
        """
        min_dist = threat_info['min_dist']

        if min_dist < self.MISSILE_LETHAL_RADIUS:
            # 致命威胁：全力逃跑
            return {
                'level': '致命',
                'speed': self.MANNED_MAX_SPEED if is_manned else self.UAV_MAX_SPEED,
                'distance': self.EVADE_DISTANCE_CRITICAL,
                'crank_angle': self.CRANK_ANGLE_LETHAL,
                'alt_change': self.ALTITUDE_CHANGE_CRITICAL
            }
        elif min_dist < self.MISSILE_CRITICAL_RADIUS:
            # 紧急威胁
            return {
                'level': '紧急',
                'speed': self.MANNED_EVADE_SPEED_CRITICAL if is_manned else self.UAV_EVADE_SPEED_CRITICAL,
                'distance': self.EVADE_DISTANCE_CRITICAL,
                'crank_angle': self.CRANK_ANGLE_CRITICAL,
                'alt_change': self.ALTITUDE_CHANGE_CRITICAL
            }
        elif min_dist < self.MISSILE_DANGER_RADIUS:
            # 危险威胁
            return {
                'level': '危险',
                'speed': self.MANNED_EVADE_SPEED_DANGER if is_manned else self.UAV_EVADE_SPEED_DANGER,
                'distance': self.EVADE_DISTANCE_DANGER,
                'crank_angle': self.CRANK_ANGLE_DANGER,
                'alt_change': self.ALTITUDE_CHANGE_DANGER
            }
        else:
            # 正常威胁
            return {
                'level': '正常',
                'speed': self.MANNED_EVADE_SPEED_NORMAL if is_manned else self.UAV_EVADE_SPEED_NORMAL,
                'distance': self.EVADE_DISTANCE_NORMAL,
                'crank_angle': self.CRANK_ANGLE_NORMAL,
                'alt_change': self.ALTITUDE_CHANGE_NORMAL
            }

    def _calculate_crank_direction(self, u_lon, u_lat, primary_threat, crank_angle, agent):
        """
        计算Cranking机动方向

        Cranking原理：
        - 不是直接逃跑（180°），而是斜向后方飞行
        - 这样可以保持一定的雷达视野，同时迫使导弹转向
        - 导弹转向会消耗能量，降低其速度和射程

        选择左Crank还是右Crank：
        - 选择离边界更远的方向
        - 选择更靠近战场中心的方向
        """
        missile_heading = primary_threat['heading']

        # 计算逃跑方向（导弹航向的反方向）
        away_dir = (missile_heading + 180) % 360

        # 计算左右Crank方向
        # Crank角度是相对于逃跑方向的偏转
        half_crank = (180 - crank_angle) / 2
        left_crank = (away_dir - half_crank) % 360
        right_crank = (away_dir + half_crank) % 360

        # 评估两个方向
        left_score = self._evaluate_evade_direction(u_lon, u_lat, left_crank, agent)
        right_score = self._evaluate_evade_direction(u_lon, u_lat, right_crank, agent)

        return left_crank if left_score >= right_score else right_crank

    def _evaluate_evade_direction(self, u_lon, u_lat, direction, agent):
        """
        评估规避方向的优劣

        评分标准：
        1. 边界安全性（不会飞出战场）
        2. 距离战场中心的远近

        返回: 评分（越高越好）
        """
        # 计算沿该方向飞行20km后的位置
        test_dist = 20  # km
        lon_off, lat_off = YxGeoUtils.km_to_lon_lat(u_lat, test_dist, direction)
        target_lon = u_lon + lon_off
        target_lat = u_lat + lat_off

        score = 100  # 基础分

        # 边界安全性检查
        bf = agent.battlefield
        if bf.get('min_lon') is not None:
            margin = self.BOUNDARY_MARGIN

            # 检查是否在安全边界内
            if not (bf['min_lon'] + margin < target_lon < bf['max_lon'] - margin and
                    bf['min_lat'] + margin < target_lat < bf['max_lat'] - margin):
                score -= 50  # 接近边界扣分

            # 计算到边界的最小距离并加分
            dist_to_min_lon = target_lon - bf['min_lon']
            dist_to_max_lon = bf['max_lon'] - target_lon
            dist_to_min_lat = target_lat - bf['min_lat']
            dist_to_max_lat = bf['max_lat'] - target_lat
            min_boundary_dist = min(dist_to_min_lon, dist_to_max_lon, dist_to_min_lat, dist_to_max_lat)
            score += min_boundary_dist * 10  # 离边界越远越好

        # 距离战场中心的评分
        dist_to_center = abs(target_lat - agent.center_lat) + abs(target_lon - agent.center_lon)
        score -= dist_to_center * 5  # 离中心越近越好（保持战场控制）

        return score

    def _calculate_evade_point(self, u_lon, u_lat, ref_lat, direction, distance_km, agent):
        """
        计算规避目标点

        确保目标点在战场边界内
        """
        lon_off, lat_off = YxGeoUtils.km_to_lon_lat(ref_lat, distance_km, direction)
        evade_lon = u_lon + lon_off
        evade_lat = u_lat + lat_off

        # 边界约束
        bf = agent.battlefield
        if bf.get('min_lon') is not None:
            margin = self.BOUNDARY_MARGIN
            evade_lon = max(bf['min_lon'] + margin, min(bf['max_lon'] - margin, evade_lon))
            evade_lat = max(bf['min_lat'] + margin, min(bf['max_lat'] - margin, evade_lat))

        return evade_lon, evade_lat

    def _calculate_evade_altitude(self, current_alt, threat_dist, alt_change, is_manned, unit_name):
        """
        计算规避高度

        高度变化策略：
        1. 紧急情况下快速变化高度
        2. 交替上升/下降避免被预测
        3. 严格遵守高度限制（2000-7000m）
        """
        min_alt = self.MIN_ALTITUDE
        max_alt = self.MAX_ALTITUDE

        # 根据单位名称决定升降方向（简单的奇偶分配）
        # 这样编队内的飞机会向不同方向分散
        go_down = unit_name.endswith(('1', '3', '5', '7', '9'))

        # 致命威胁时优先下降（利用地面杂波干扰导弹）
        if threat_dist < self.MISSILE_LETHAL_RADIUS:
            target_alt = current_alt - alt_change
        elif go_down:
            target_alt = current_alt - alt_change
        else:
            target_alt = current_alt + alt_change

        # 应用高度限制
        target_alt = max(min_alt, min(max_alt, target_alt))

        return target_alt

    def _check_boundary_safe(self, lon, lat, agent):
        """检查坐标是否在安全边界内"""
        bf = agent.battlefield
        if bf.get('min_lon') is None:
            return True
        margin = self.BOUNDARY_MARGIN
        return (bf['min_lon'] + margin < lon < bf['max_lon'] - margin and
                bf['min_lat'] + margin < lat < bf['max_lat'] - margin)


class ActionEvadeMissilesAdvanced(ActionEvadeMissiles):
    """
    高级导弹规避策略

    在基础规避策略上增加：
    1. 多导弹威胁综合评估
    2. 编队协同规避
    3. 诱饵战术（无人机为有人机挡弹）
    """

    # 有人机保护优先级
    PROTECT_MANNED_PRIORITY = True

    # 无人机诱饵距离：无人机在有人机前方多远吸引导弹
    DECOY_DISTANCE = 5000  # m

    def _process_unit_evasion(self, agent, unit, all_missiles):
        """
        增强版规避处理

        对于有人机：优先保护
        对于无人机：可以作为诱饵
        """
        is_manned = unit.get('type') == '有人机'

        if is_manned and self.PROTECT_MANNED_PRIORITY:
            # 有人机使用更保守的规避策略
            return self._process_manned_evasion(agent, unit, all_missiles)
        else:
            # 无人机使用标准规避
            return super()._process_unit_evasion(agent, unit, all_missiles)

    def _process_manned_evasion(self, agent, unit, all_missiles):
        """
        有人机专用规避策略

        特点：
        1. 更早开始规避（威胁半径增大20%）
        2. 更激进的规避角度
        3. 优先保持在无人机后方
        """
        # 增大有人机的威胁感知半径
        original_threat_radius = self.MISSILE_THREAT_RADIUS
        self.MISSILE_THREAT_RADIUS = int(original_threat_radius * 1.2)

        result = super()._process_unit_evasion(agent, unit, all_missiles)

        # 恢复原始参数
        self.MISSILE_THREAT_RADIUS = original_threat_radius

        return result
