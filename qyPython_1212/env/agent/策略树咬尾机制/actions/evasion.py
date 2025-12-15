"""
导弹规避动作节点 - 垂直躲避策略 + 导弹路径建模

基于官方赛题参数设计的垂直规避策略：
- 有人机：最大速度500m/s，高度范围2000-7000m
- 无人机：最大速度360m/s，高度范围2000-7000m
- 导弹：速度1200m/s，最大飞行时间60s，最大飞行距离72km

垂直躲避策略特点：
- 检测到导弹威胁时，主要通过快速改变高度来规避
- 利用导弹在垂直方向追踪能力较弱的特点
- 高度变化范围: 2000-7000m (5000m可用空间)
- 配合水平方向小幅度机动

新增：导弹路径建模与垂直躲避系统
- 将导弹飞行路径建模为射线（起点+方向）
- 计算飞机到导弹路径的垂直距离
- 如果飞机在导弹路径上，立即全速垂直远离
"""

import math
from typing import Dict, List
from ..bt_framework import Action, NodeStatus
from utilities.yxScriptTreeFunc import YxScriptTreeFunc as decCmd
from utilities.yxGeoUtils import YxGeoUtils


# ============================================================================
# 导弹路径建模与垂直躲避系统
# ============================================================================

class MissilePathModel:
    """
    导弹路径建模 - 将导弹飞行路径建模为射线

    用于：
    1. 计算飞机到导弹路径的垂直距离
    2. 判断飞机是否在导弹路径上
    3. 计算最优垂直逃逸方向

    数学原理：
    - 导弹路径建模为射线：P(t) = M + t * D, t ≥ 0
    - 点到直线距离：d = |MA × D| / |D|
    """

    # 安全距离阈值
    DANGER_DISTANCE = 2000      # 2km内算在导弹路径上，需要立即躲避
    WARNING_DISTANCE = 5000     # 5km内算接近导弹路径，需要预警
    ESCAPE_DISTANCE_KM = 15     # 垂直逃逸飞行距离 km

    def __init__(self, missile: Dict):
        """
        初始化导弹路径模型

        Args:
            missile: 导弹数据字典，包含 longitude, latitude, heading
        """
        self.origin_lon = missile.get('longitude', 0)
        self.origin_lat = missile.get('latitude', 0)
        self.heading = math.degrees(missile.get('heading', 0)) % 360
        self.speed = missile.get('speed', 1200)  # 默认1200m/s
        self.missile_id = missile.get('target_id') or missile.get('name') or id(missile)

        # 计算方向向量（归一化）
        heading_rad = math.radians(self.heading)
        self.dir_x = math.sin(heading_rad)  # 东向分量
        self.dir_y = math.cos(heading_rad)  # 北向分量

    def get_perpendicular_distance(self, unit_lon: float, unit_lat: float) -> float:
        """
        计算飞机到导弹路径的垂直距离

        使用向量叉积法：d = |MA × D| / |D|

        Returns:
            垂直距离（米）
        """
        # 将经纬度转换为相对距离（米）
        # MA向量：从导弹位置到飞机位置
        delta_lon = unit_lon - self.origin_lon
        delta_lat = unit_lat - self.origin_lat

        # 经纬度转米（近似）
        lat_to_m = 111320  # 1度纬度约111km
        lon_to_m = 111320 * math.cos(math.radians(self.origin_lat))

        ma_x = delta_lon * lon_to_m  # 东向距离
        ma_y = delta_lat * lat_to_m  # 北向距离

        # 叉积 |MA × D| = |ma_x * dir_y - ma_y * dir_x|
        cross_product = abs(ma_x * self.dir_y - ma_y * self.dir_x)

        # |D| = 1（已归一化）
        return cross_product

    def get_perpendicular_escape_direction(self, unit_lon: float, unit_lat: float) -> float:
        """
        计算垂直于导弹路径的最优逃逸方向

        选择远离导弹当前位置的垂直方向

        Returns:
            逃逸航向（度，0-360）
        """
        # 两个垂直方向
        perp_left = (self.heading - 90) % 360
        perp_right = (self.heading + 90) % 360

        # 计算飞机相对导弹的位置
        delta_lon = unit_lon - self.origin_lon
        delta_lat = unit_lat - self.origin_lat

        # 计算飞机在导弹路径哪一侧
        # 使用叉积的符号判断
        lat_to_m = 111320
        lon_to_m = 111320 * math.cos(math.radians(self.origin_lat))

        ma_x = delta_lon * lon_to_m
        ma_y = delta_lat * lat_to_m

        cross = ma_x * self.dir_y - ma_y * self.dir_x

        # 正值表示在右侧，负值表示在左侧
        # 选择继续远离的方向
        if cross >= 0:
            return perp_right  # 飞机在右侧，继续向右逃
        else:
            return perp_left   # 飞机在左侧，继续向左逃

    def is_aircraft_on_path(self, unit_lon: float, unit_lat: float) -> bool:
        """判断飞机是否在导弹路径上（危险区域内）"""
        return self.get_perpendicular_distance(unit_lon, unit_lat) < self.DANGER_DISTANCE

    def is_aircraft_approaching_path(self, unit_lon: float, unit_lat: float,
                                      unit_heading: float) -> bool:
        """
        判断飞机航向是否会接近导弹路径

        计算飞机当前航向与导弹路径的交点，
        如果会在未来某时刻进入危险区域，返回True
        """
        perp_dist = self.get_perpendicular_distance(unit_lon, unit_lat)

        if perp_dist > self.WARNING_DISTANCE:
            return False  # 已经足够远，不用担心

        # 计算飞机航向与导弹路径的夹角
        angle_diff = abs((unit_heading - self.heading + 180) % 360 - 180)

        # 如果飞机航向与导弹路径近似平行或相向，且在警告距离内
        if angle_diff < 30 or angle_diff > 150:
            # 检查是否在向导弹路径靠近
            escape_dir = self.get_perpendicular_escape_direction(unit_lon, unit_lat)
            heading_to_escape_diff = abs((unit_heading - escape_dir + 180) % 360 - 180)

            # 如果航向偏离逃逸方向超过90度，说明在靠近
            if heading_to_escape_diff > 90:
                return True

        return False


class MissilePathManager:
    """
    导弹路径管理器 - 管理所有活跃导弹的路径模型

    功能：
    1. 维护所有导弹的路径模型
    2. 检查飞机是否在任何导弹路径上
    3. 计算最优躲避方向
    """

    def __init__(self):
        self.missile_paths = {}  # missile_id -> MissilePathModel

    def update_missiles(self, missiles: List[Dict]):
        """更新导弹路径模型"""
        current_ids = set()

        for missile in missiles:
            m_id = missile.get('target_id') or missile.get('name') or id(missile)
            current_ids.add(m_id)

            # 创建或更新路径模型
            self.missile_paths[m_id] = MissilePathModel(missile)

        # 移除已消失的导弹
        expired_ids = set(self.missile_paths.keys()) - current_ids
        for m_id in expired_ids:
            del self.missile_paths[m_id]

    def check_unit_danger(self, unit: Dict) -> Dict:
        """
        检查飞机是否在任何导弹路径上

        Returns:
            {
                'in_danger': bool,          # 是否在危险区域
                'approaching': bool,        # 是否在接近危险区域
                'closest_distance': float,  # 到最近导弹路径的距离
                'escape_direction': float,  # 推荐逃逸方向
                'threatening_missiles': []  # 威胁导弹列表
            }
        """
        u_lon = unit.get('longitude', 0)
        u_lat = unit.get('latitude', 0)
        u_heading = math.degrees(unit.get('heading', 0)) % 360

        result = {
            'in_danger': False,
            'approaching': False,
            'closest_distance': float('inf'),
            'escape_direction': None,
            'threatening_missiles': []
        }

        for m_id, path in self.missile_paths.items():
            perp_dist = path.get_perpendicular_distance(u_lon, u_lat)

            if perp_dist < result['closest_distance']:
                result['closest_distance'] = perp_dist
                result['escape_direction'] = path.get_perpendicular_escape_direction(u_lon, u_lat)

            if path.is_aircraft_on_path(u_lon, u_lat):
                result['in_danger'] = True
                result['threatening_missiles'].append(m_id)
            elif path.is_aircraft_approaching_path(u_lon, u_lat, u_heading):
                result['approaching'] = True
                result['threatening_missiles'].append(m_id)

        return result


# ============================================================================
# 原有躲避动作节点
# ============================================================================

class ActionEvadeMissiles(Action):
    """
    垂直躲避策略 - 主要通过高度变化规避导弹

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

    垂直躲避原理：
    1. 导弹在垂直方向的追踪能力相对较弱
    2. 快速改变高度可以使导弹能量消耗增大
    3. 利用高度差增加导弹飞行距离
    4. 在高度边界附近进行往复机动
    """

    # ========== 官方飞机参数 ==========
    MANNED_MIN_SPEED = 180          # 有人机最小速度 m/s
    MANNED_MAX_SPEED = 500          # 有人机最大速度 m/s
    MANNED_MIN_ALT = 2000           # 有人机最低高度 m
    MANNED_MAX_ALT = 7000           # 有人机最高高度 m

    UAV_MIN_SPEED = 120             # 无人机最小速度 m/s
    UAV_MAX_SPEED = 360             # 无人机最大速度 m/s
    UAV_MIN_ALT = 2000              # 无人机最低高度 m
    UAV_MAX_ALT = 7000              # 无人机最高高度 m

    # ========== 官方导弹参数 ==========
    MISSILE_SPEED = 1200            # 导弹平均速度 m/s
    MISSILE_MAX_FLIGHT_TIME = 60    # 导弹最长飞行时间 s
    MISSILE_MAX_RANGE = 72000       # 导弹最大飞行距离 m (72km)

    # ========== 官方战场参数 ==========
    BATTLEFIELD_SIZE = 200000       # 任务区域大小 m (200km)
    INITIAL_ALTITUDE = 4000         # 初始部署高度 m

    # ========== 威胁判断参数 ==========
    MISSILE_THREAT_RADIUS = 50000       # 威胁半径50km（开始关注）
    MISSILE_DANGER_RADIUS = 35000       # 危险半径35km（积极规避）
    MISSILE_CRITICAL_RADIUS = 25000     # 紧急半径25km（全力规避）
    MISSILE_LETHAL_RADIUS = 12000       # 致命半径12km（极限规避）
    MISSILE_THREAT_ANGLE = 75           # 导弹威胁角度

    # ========== 垂直躲避参数 ==========
    # 高度限制（官方参数）
    MIN_ALTITUDE = 2000                 # 最低飞行高度 m
    MAX_ALTITUDE = 7000                 # 最高飞行高度 m
    MID_ALTITUDE = 4500                 # 中间高度 m

    # 垂直躲避高度变化量（根据威胁等级）
    VERTICAL_EVADE_NORMAL = 1500        # 正常威胁：变化1500m
    VERTICAL_EVADE_DANGER = 2500        # 危险威胁：变化2500m
    VERTICAL_EVADE_CRITICAL = 3500      # 紧急威胁：变化3500m
    VERTICAL_EVADE_LETHAL = 4500        # 致命威胁：变化4500m（几乎到边界）

    # 规避速度（使用最大速度）
    MANNED_EVADE_SPEED = 500            # 有人机规避速度
    UAV_EVADE_SPEED = 360               # 无人机规避速度

    # 水平方向小幅度偏移（配合垂直机动）
    HORIZONTAL_OFFSET_KM = 2            # 水平偏移距离 km

    # ========== 边界安全参数 ==========
    BOUNDARY_MARGIN = 0.09              # 边界安全边距（经纬度，约10km）

    # ========== 调试参数 ==========
    DEBUG_ENABLED = False               # 调试开关
    DEBUG_INTERVAL = 20                 # 调试输出间隔帧数

    def tick(self, agent) -> str:
        """
        执行垂直躲避逻辑

        优先级：
        1. 检测所有来袭导弹
        2. 评估每个己方单位的威胁等级
        3. 计算垂直规避方向（上升/下降）
        4. 执行垂直规避机动
        """
        if self.DEBUG_ENABLED:
            if not hasattr(agent, '_missile_debug_frame'):
                agent._missile_debug_frame = 0
            if agent.frame_count - agent._missile_debug_frame >= self.DEBUG_INTERVAL:
                agent._missile_debug_frame = agent.frame_count
                print(f"[垂直躲避] Frame {agent.frame_count}: 检测到 {len(agent.enemy_missiles)} 枚敌方导弹")

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
            if self._process_vertical_evasion(agent, unit, all_missiles):
                evade_count += 1

        if self.DEBUG_ENABLED and evade_count > 0:
            print(f"[垂直躲避] Frame {agent.frame_count}: {evade_count} 个单位执行垂直规避")

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

    def _process_vertical_evasion(self, agent, unit, all_missiles):
        """
        处理单个单位的垂直躲避逻辑

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
        threat_info = self._analyze_threats(u_lon, u_lat, u_alt, all_missiles)

        if not threat_info['threats']:
            return False

        # 确定威胁等级和垂直规避参数
        evade_params = self._get_vertical_evade_params(threat_info, u_alt, is_manned)

        # 计算目标高度（垂直躲避核心）
        target_alt = self._calculate_vertical_evade_altitude(
            u_alt,
            threat_info['primary_threat'],
            evade_params['alt_change'],
            is_manned,
            unit_name
        )

        # 计算小幅度水平偏移（配合垂直机动）
        evade_lon, evade_lat = self._calculate_horizontal_offset(
            u_lon, u_lat,
            threat_info['primary_threat'],
            agent
        )

        # 执行垂直躲避
        speed = self.MANNED_EVADE_SPEED if is_manned else self.UAV_EVADE_SPEED
        agent.add_action(
            decCmd.fly_to_point(unit_name, (evade_lat, evade_lon, target_alt), speed),
            unit_name
        )

        # 调试输出
        if self.DEBUG_ENABLED:
            level = evade_params['level']
            direction = "上升" if target_alt > u_alt else "下降"
            alt_change = abs(target_alt - u_alt)
            print(f"[垂直躲避] [{level}] {unit_name}: "
                  f"距离={threat_info['min_dist']:.0f}m, "
                  f"{direction}{alt_change:.0f}m, "
                  f"目标高度={target_alt:.0f}m")

        return True

    def _analyze_threats(self, u_lon, u_lat, u_alt, all_missiles):
        """
        分析导弹威胁（包含高度信息）

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
            # 计算三维距离
            horizontal_dist = YxGeoUtils.haversine_distance(u_lon, u_lat, missile['lon'], missile['lat'])
            vertical_dist = abs(u_alt - missile['altitude'])
            dist_3d = math.sqrt(horizontal_dist**2 + vertical_dist**2)

            # 超出威胁半径则忽略
            if horizontal_dist > self.MISSILE_THREAT_RADIUS:
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
                    'dist': horizontal_dist,
                    'dist_3d': dist_3d,
                    'vertical_dist': vertical_dist,
                    'bearing_to_unit': bearing_to_unit,
                    'angle_diff': angle_diff
                }
                threats.append(threat)

                if horizontal_dist < min_dist:
                    min_dist = horizontal_dist
                    primary_threat = threat

        return {
            'threats': threats,
            'primary_threat': primary_threat,
            'min_dist': min_dist if threats else float('inf')
        }

    def _get_vertical_evade_params(self, threat_info, current_alt, is_manned):
        """
        根据威胁等级获取垂直规避参数

        威胁等级：
        - LETHAL: < 12km (致命) -> 最大高度变化
        - CRITICAL: < 25km (紧急) -> 大幅高度变化
        - DANGER: < 35km (危险) -> 中等高度变化
        - NORMAL: < 50km (正常) -> 小幅高度变化
        """
        min_dist = threat_info['min_dist']

        if min_dist < self.MISSILE_LETHAL_RADIUS:
            return {
                'level': '致命',
                'alt_change': self.VERTICAL_EVADE_LETHAL
            }
        elif min_dist < self.MISSILE_CRITICAL_RADIUS:
            return {
                'level': '紧急',
                'alt_change': self.VERTICAL_EVADE_CRITICAL
            }
        elif min_dist < self.MISSILE_DANGER_RADIUS:
            return {
                'level': '危险',
                'alt_change': self.VERTICAL_EVADE_DANGER
            }
        else:
            return {
                'level': '正常',
                'alt_change': self.VERTICAL_EVADE_NORMAL
            }

    def _calculate_vertical_evade_altitude(self, current_alt, primary_threat, alt_change, is_manned, unit_name):
        """
        计算垂直躲避目标高度

        策略：
        1. 如果导弹在我方上方，则下降躲避
        2. 如果导弹在我方下方，则上升躲避
        3. 如果导弹高度相近，根据当前高度选择最大变化方向
        4. 优先利用高度边界（2000m或7000m）
        """
        min_alt = self.MIN_ALTITUDE
        max_alt = self.MAX_ALTITUDE
        missile_alt = primary_threat['altitude']

        # 计算与导弹的高度差
        alt_diff = current_alt - missile_alt

        # 计算上升和下降的可用空间
        space_up = max_alt - current_alt
        space_down = current_alt - min_alt

        # 决定躲避方向
        if abs(alt_diff) > 500:
            # 导弹明显不在同一高度，向反方向躲避
            if alt_diff > 0:
                # 我方在上，导弹在下，继续上升
                target_alt = current_alt + alt_change
            else:
                # 我方在下，导弹在上，继续下降
                target_alt = current_alt - alt_change
        else:
            # 导弹高度相近，选择可用空间更大的方向
            if space_up >= space_down:
                # 上方空间大，上升
                target_alt = current_alt + alt_change
            else:
                # 下方空间大，下降
                target_alt = current_alt - alt_change

        # 应用高度限制
        target_alt = max(min_alt, min(max_alt, target_alt))

        # 如果接近边界，尝试往复机动
        if target_alt <= min_alt + 200:
            # 已到下边界，下次会上升
            target_alt = min_alt
        elif target_alt >= max_alt - 200:
            # 已到上边界，下次会下降
            target_alt = max_alt

        return target_alt

    def _calculate_horizontal_offset(self, u_lon, u_lat, primary_threat, agent):
        """
        计算水平方向小幅度偏移（配合垂直机动）

        向导弹来袭方向的垂直方向偏移，增加导弹追踪难度
        """
        missile_heading = primary_threat['heading']

        # 计算垂直于导弹航向的方向（左或右偏90度）
        perpendicular_left = (missile_heading - 90) % 360
        perpendicular_right = (missile_heading + 90) % 360

        # 选择离边界更远的方向
        left_score = self._evaluate_direction(u_lon, u_lat, perpendicular_left, agent)
        right_score = self._evaluate_direction(u_lon, u_lat, perpendicular_right, agent)

        offset_direction = perpendicular_left if left_score >= right_score else perpendicular_right

        # 计算偏移后的位置
        lon_off, lat_off = YxGeoUtils.km_to_lon_lat(u_lat, self.HORIZONTAL_OFFSET_KM, offset_direction)
        evade_lon = u_lon + lon_off
        evade_lat = u_lat + lat_off

        # 边界约束
        bf = agent.battlefield
        if bf.get('min_lon') is not None:
            margin = self.BOUNDARY_MARGIN
            evade_lon = max(bf['min_lon'] + margin, min(bf['max_lon'] - margin, evade_lon))
            evade_lat = max(bf['min_lat'] + margin, min(bf['max_lat'] - margin, evade_lat))

        return evade_lon, evade_lat

    def _evaluate_direction(self, u_lon, u_lat, direction, agent):
        """评估水平偏移方向的优劣"""
        test_dist = 5  # km
        lon_off, lat_off = YxGeoUtils.km_to_lon_lat(u_lat, test_dist, direction)
        target_lon = u_lon + lon_off
        target_lat = u_lat + lat_off

        score = 100

        bf = agent.battlefield
        if bf.get('min_lon') is not None:
            margin = self.BOUNDARY_MARGIN
            if not (bf['min_lon'] + margin < target_lon < bf['max_lon'] - margin and
                    bf['min_lat'] + margin < target_lat < bf['max_lat'] - margin):
                score -= 50

        return score


class ActionEvadeMissilesAdvanced(ActionEvadeMissiles):
    """
    高级垂直躲避策略

    在基础垂直躲避策略上增加：
    1. 有人机优先保护（更早开始规避）
    2. 编队高度分层（避免碰撞）
    """

    PROTECT_MANNED_PRIORITY = True

    def _process_vertical_evasion(self, agent, unit, all_missiles):
        """
        增强版垂直躲避处理

        对于有人机：更早开始规避
        """
        is_manned = unit.get('type') == '有人机'

        if is_manned and self.PROTECT_MANNED_PRIORITY:
            return self._process_manned_vertical_evasion(agent, unit, all_missiles)
        else:
            return super()._process_vertical_evasion(agent, unit, all_missiles)

    def _process_manned_vertical_evasion(self, agent, unit, all_missiles):
        """
        有人机专用垂直躲避策略

        特点：
        1. 更早开始规避（威胁半径增大20%）
        2. 更大的高度变化量
        """
        original_threat_radius = self.MISSILE_THREAT_RADIUS
        self.MISSILE_THREAT_RADIUS = int(original_threat_radius * 1.2)

        result = super()._process_vertical_evasion(agent, unit, all_missiles)

        self.MISSILE_THREAT_RADIUS = original_threat_radius

        return result


class ActionTacticalEvasion(ActionEvadeMissiles):
    """
    战术躲避 - 基于导弹路径建模的垂直躲避

    核心功能（新增）：
    1. 使用 MissilePathModel 将导弹路径建模为射线
    2. 计算飞机到导弹路径的垂直距离
    3. 如果飞机在导弹路径上（<2km），立即全速垂直逃逸
    4. 如果飞机正在接近导弹路径（<5km），调整航向远离

    原有功能：
    1. 检测1-4发来袭导弹
    2. 计算所有导弹的综合威胁方向和高度
    3. 计算最优垂直躲避角度，尽可能实现垂直躲避
    4. 配合小幅度水平机动增加规避效果
    """

    # ========== 路径管理器（类级别单例）==========
    path_manager = MissilePathManager()

    # ========== 多导弹检测参数 ==========
    MULTI_MISSILE_DETECTION_RANGE = 50000   # 检测范围 (50km)
    MAX_MISSILES_TO_TRACK = 4               # 最多跟踪4发导弹
    MIN_MISSILES_FOR_TACTICAL = 1           # 至少1发导弹触发战术躲避

    # ========== 战术垂直躲避参数 ==========
    # 根据导弹数量调整高度变化量
    ALTITUDE_CHANGE_1_MISSILE = 2500        # 1发导弹：变化2500m
    ALTITUDE_CHANGE_2_MISSILES = 3500       # 2发导弹：变化3500m
    ALTITUDE_CHANGE_3_MISSILES = 4000       # 3发导弹：变化4000m
    ALTITUDE_CHANGE_4_MISSILES = 4500       # 4发导弹：变化4500m（极限）

    TACTICAL_SPEED_MANNED = 500             # 有人机战术机动速度
    TACTICAL_SPEED_UAV = 360                # 无人机战术机动速度

    # ========== 综合躲避角度计算参数 ==========
    # 水平偏移距离（配合垂直机动）
    HORIZONTAL_EVADE_DISTANCE_KM = 3        # 水平躲避距离 km

    # 威胁权重：距离越近权重越大
    DISTANCE_WEIGHT_FACTOR = 1.5            # 距离权重因子

    # ========== 路径躲避参数 ==========
    PATH_ESCAPE_DISTANCE_KM = 15            # 垂直逃逸飞行距离 km
    PATH_ADJUST_DISTANCE_KM = 8             # 航向调整飞行距离 km

    DEBUG_TACTICAL = False

    def tick(self, agent) -> str:
        """
        执行导弹躲避（优先使用路径建模）

        优先级：
        1. 更新导弹路径模型
        2. 检查是否在导弹路径上 -> 全速垂直逃逸
        3. 检查是否正在接近导弹路径 -> 调整航向
        4. 否则使用原有多导弹威胁分析
        """
        # 无导弹则直接返回
        if not agent.enemy_missiles:
            return NodeStatus.SUCCESS

        # 更新导弹路径模型
        self.path_manager.update_missiles(agent.enemy_missiles)

        # 对每个己方单位检查危险
        for unit in agent.own_units:
            # 首先使用路径建模检查
            danger_info = self.path_manager.check_unit_danger(unit)

            if danger_info['in_danger']:
                # 在导弹路径上！立即全速垂直逃逸（最高优先级）
                self._execute_perpendicular_escape(agent, unit, danger_info)
            elif danger_info['approaching']:
                # 正在接近导弹路径，调整航向
                self._execute_heading_adjustment(agent, unit, danger_info)
            else:
                # 安全，使用原有多导弹威胁分析
                missile_threat = self._detect_multi_missile_threat(agent, unit)
                if missile_threat['has_threat']:
                    self._execute_multi_missile_vertical_evasion(agent, unit, missile_threat)

        return NodeStatus.SUCCESS

    def _execute_perpendicular_escape(self, agent, unit: Dict, danger_info: Dict):
        """
        执行垂直逃逸 - 全速垂直于导弹路径飞行

        当飞机在导弹路径上（距离<2km）时触发
        立即以最大速度向垂直于导弹路径的方向飞行15km
        """
        unit_name = unit['name']
        u_lon = unit.get('longitude', 0)
        u_lat = unit.get('latitude', 0)
        u_alt = unit.get('altitude', self.INITIAL_ALTITUDE)
        is_manned = unit.get('type') == '有人机'

        # 获取逃逸方向
        escape_heading = danger_info['escape_direction']
        if escape_heading is None:
            return

        # 计算目标点（垂直方向飞行15km）
        escape_dist_km = self.PATH_ESCAPE_DISTANCE_KM
        lon_off, lat_off = YxGeoUtils.km_to_lon_lat(u_lat, escape_dist_km, escape_heading)
        target_lon = u_lon + lon_off
        target_lat = u_lat + lat_off

        # 边界约束
        bf = agent.battlefield
        if bf.get('min_lon') is not None:
            margin = self.BOUNDARY_MARGIN
            target_lon = max(bf['min_lon'] + margin, min(bf['max_lon'] - margin, target_lon))
            target_lat = max(bf['min_lat'] + margin, min(bf['max_lat'] - margin, target_lat))

        # 全速飞行
        max_speed = self.TACTICAL_SPEED_MANNED if is_manned else self.TACTICAL_SPEED_UAV

        agent.add_action(
            decCmd.fly_to_point(unit_name, (target_lat, target_lon, u_alt), max_speed),
            unit_name
        )

        if self.DEBUG_TACTICAL or self.DEBUG_ENABLED:
            print(f"[垂直逃逸] {unit_name}")
            print(f"  到路径距离: {danger_info['closest_distance']:.0f}m")
            print(f"  逃逸方向: {escape_heading:.1f}°")
            print(f"  威胁导弹: {danger_info['threatening_missiles']}")

    def _execute_heading_adjustment(self, agent, unit: Dict, danger_info: Dict):
        """
        执行航向调整 - 飞机航向正在接近导弹路径时调整

        当飞机正在接近导弹路径（距离2-5km）时触发
        调整航向向远离导弹路径的方向飞行8km
        """
        unit_name = unit['name']
        u_lon = unit.get('longitude', 0)
        u_lat = unit.get('latitude', 0)
        u_alt = unit.get('altitude', self.INITIAL_ALTITUDE)
        is_manned = unit.get('type') == '有人机'

        # 调整航向远离导弹路径
        safe_heading = danger_info['escape_direction']
        if safe_heading is None:
            return

        # 计算目标点（调整航向方向飞行8km）
        adjust_dist_km = self.PATH_ADJUST_DISTANCE_KM
        lon_off, lat_off = YxGeoUtils.km_to_lon_lat(u_lat, adjust_dist_km, safe_heading)
        target_lon = u_lon + lon_off
        target_lat = u_lat + lat_off

        # 边界约束
        bf = agent.battlefield
        if bf.get('min_lon') is not None:
            margin = self.BOUNDARY_MARGIN
            target_lon = max(bf['min_lon'] + margin, min(bf['max_lon'] - margin, target_lon))
            target_lat = max(bf['min_lat'] + margin, min(bf['max_lat'] - margin, target_lat))

        # 较高速度调整
        speed = 450 if is_manned else 320

        agent.add_action(
            decCmd.fly_to_point(unit_name, (target_lat, target_lon, u_alt), speed),
            unit_name
        )

        if self.DEBUG_TACTICAL or self.DEBUG_ENABLED:
            print(f"[航向调整] {unit_name}")
            print(f"  到路径距离: {danger_info['closest_distance']:.0f}m")
            print(f"  调整航向: {safe_heading:.1f}°")

    def _detect_multi_missile_threat(self, agent, unit):
        """
        检测多导弹威胁（1-4发）

        分析所有来袭导弹，计算综合威胁信息

        返回: {
            'has_threat': bool,
            'missiles': 威胁导弹列表（最多4发，按距离排序）,
            'missile_count': 导弹数量,
            'weighted_avg_altitude': 加权平均高度,
            'weighted_avg_bearing': 加权平均来袭方向,
            'min_distance': 最近导弹距离,
            'composite_evade_direction': 综合躲避方向
        }
        """
        unit_name = unit['name']
        u_lon = unit.get('longitude', 0)
        u_lat = unit.get('latitude', 0)
        u_alt = unit.get('altitude', self.INITIAL_ALTITUDE)

        if not u_lon or not u_lat:
            return {'has_threat': False}

        # 收集威胁导弹
        threatening_missiles = []

        for missile in agent.enemy_missiles:
            m_lon = missile.get('longitude', 0)
            m_lat = missile.get('latitude', 0)
            m_heading = math.degrees(missile.get('heading', 0)) % 360
            m_alt = missile.get('altitude', self.INITIAL_ALTITUDE)
            m_speed = missile.get('speed', self.MISSILE_SPEED)

            if not m_lon or not m_lat:
                continue

            # 计算距离
            dist = YxGeoUtils.haversine_distance(u_lon, u_lat, m_lon, m_lat)

            if dist > self.MULTI_MISSILE_DETECTION_RANGE:
                continue

            # 计算导弹是否指向我方
            bearing_to_me = YxGeoUtils.calculate_bearing(m_lon, m_lat, u_lon, u_lat)
            targeting_angle = abs((bearing_to_me - m_heading + 180) % 360 - 180)

            # 只考虑指向我方的导弹（威胁角度内）
            if targeting_angle < self.MISSILE_THREAT_ANGLE:
                # 计算导弹相对我方的来袭方向
                bearing_from_me = YxGeoUtils.calculate_bearing(u_lon, u_lat, m_lon, m_lat)

                threatening_missiles.append({
                    'lon': m_lon,
                    'lat': m_lat,
                    'altitude': m_alt,
                    'heading': m_heading,
                    'speed': m_speed,
                    'distance': dist,
                    'bearing_from_me': bearing_from_me,
                    'targeting_angle': targeting_angle,
                    'id': missile.get('target_id') or missile.get('name') or id(missile)
                })

        if len(threatening_missiles) < self.MIN_MISSILES_FOR_TACTICAL:
            return {'has_threat': False}

        # 按距离排序，只保留最近的4发
        threatening_missiles.sort(key=lambda x: x['distance'])
        threatening_missiles = threatening_missiles[:self.MAX_MISSILES_TO_TRACK]

        missile_count = len(threatening_missiles)
        min_distance = threatening_missiles[0]['distance']

        # 计算加权平均高度和来袭方向
        # 权重：距离越近权重越大 (weight = 1 / distance^factor)
        total_weight = 0
        weighted_alt_sum = 0
        weighted_bearing_x = 0  # 用于向量平均
        weighted_bearing_y = 0

        for m in threatening_missiles:
            # 距离权重（距离越近权重越大）
            weight = 1.0 / (m['distance'] ** self.DISTANCE_WEIGHT_FACTOR)
            total_weight += weight

            # 加权高度
            weighted_alt_sum += m['altitude'] * weight

            # 加权方向（使用向量平均避免角度跨越问题）
            bearing_rad = math.radians(m['bearing_from_me'])
            weighted_bearing_x += math.cos(bearing_rad) * weight
            weighted_bearing_y += math.sin(bearing_rad) * weight

        weighted_avg_altitude = weighted_alt_sum / total_weight
        weighted_avg_bearing = math.degrees(math.atan2(weighted_bearing_y, weighted_bearing_x)) % 360

        # 计算综合躲避方向（导弹来袭方向的反方向，垂直偏移90度）
        # 主要靠垂直躲避，水平方向只做小幅度偏移
        evade_horizontal_dir = (weighted_avg_bearing + 180) % 360  # 反方向
        # 选择左偏或右偏90度，远离导弹
        evade_perpendicular_left = (evade_horizontal_dir - 90) % 360
        evade_perpendicular_right = (evade_horizontal_dir + 90) % 360

        if self.DEBUG_TACTICAL:
            print(f"[多导弹躲避] {unit_name} 检测到 {missile_count} 发导弹威胁!")
            for i, m in enumerate(threatening_missiles):
                print(f"  - 导弹{i+1}: 距离={m['distance']/1000:.1f}km, "
                      f"高度={m['altitude']:.0f}m, 来袭方向={m['bearing_from_me']:.1f}°")
            print(f"  - 加权平均高度: {weighted_avg_altitude:.0f}m")
            print(f"  - 加权平均来袭方向: {weighted_avg_bearing:.1f}°")

        return {
            'has_threat': True,
            'missiles': threatening_missiles,
            'missile_count': missile_count,
            'weighted_avg_altitude': weighted_avg_altitude,
            'weighted_avg_bearing': weighted_avg_bearing,
            'min_distance': min_distance,
            'evade_perpendicular_left': evade_perpendicular_left,
            'evade_perpendicular_right': evade_perpendicular_right,
            'my_altitude': u_alt
        }

    def _execute_multi_missile_vertical_evasion(self, agent, unit, threat_info):
        """
        执行多导弹综合垂直躲避

        策略：
        1. 根据导弹数量决定高度变化量
        2. 根据导弹加权平均高度决定躲避方向（上升/下降）
        3. 计算最优垂直躲避高度
        4. 配合小幅度水平偏移
        """
        unit_name = unit['name']
        u_lon = unit.get('longitude', 0)
        u_lat = unit.get('latitude', 0)
        u_alt = unit.get('altitude', self.INITIAL_ALTITUDE)
        is_manned = unit.get('type') == '有人机'

        missile_count = threat_info['missile_count']
        weighted_avg_alt = threat_info['weighted_avg_altitude']
        min_distance = threat_info['min_distance']

        # 根据导弹数量决定高度变化量
        if missile_count >= 4:
            alt_change = self.ALTITUDE_CHANGE_4_MISSILES
        elif missile_count >= 3:
            alt_change = self.ALTITUDE_CHANGE_3_MISSILES
        elif missile_count >= 2:
            alt_change = self.ALTITUDE_CHANGE_2_MISSILES
        else:
            alt_change = self.ALTITUDE_CHANGE_1_MISSILE

        # 根据距离调整高度变化量（距离越近变化越大）
        if min_distance < self.MISSILE_LETHAL_RADIUS:
            alt_change = min(self.MAX_ALTITUDE - self.MIN_ALTITUDE, alt_change * 1.2)
        elif min_distance < self.MISSILE_CRITICAL_RADIUS:
            alt_change = alt_change * 1.1

        # 决定垂直躲避方向
        target_alt, direction = self._calculate_optimal_vertical_evade(
            u_alt, weighted_avg_alt, alt_change, threat_info['missiles']
        )

        # 计算水平偏移方向和位置
        evade_lon, evade_lat = self._calculate_composite_horizontal_offset(
            u_lon, u_lat, threat_info, agent
        )

        speed = self.TACTICAL_SPEED_MANNED if is_manned else self.TACTICAL_SPEED_UAV

        # 执行综合垂直躲避
        agent.add_action(
            decCmd.fly_to_point(unit_name, (evade_lat, evade_lon, target_alt), speed),
            unit_name
        )

        if self.DEBUG_TACTICAL:
            print(f"[多导弹躲避] {unit_name} 执行{direction}")
            print(f"  - 当前高度: {u_alt:.0f}m -> 目标高度: {target_alt:.0f}m")
            print(f"  - 高度变化: {abs(target_alt - u_alt):.0f}m")
            print(f"  - 导弹数量: {missile_count}, 最近距离: {min_distance/1000:.1f}km")

    def _calculate_optimal_vertical_evade(self, current_alt, weighted_avg_missile_alt, alt_change, missiles):
        """
        计算最优垂直躲避高度

        策略：
        1. 分析所有导弹的高度分布
        2. 选择远离导弹密集区域的方向
        3. 优先利用高度边界进行极限规避
        4. 考虑上升/下降空间选择最优方向

        返回: (target_altitude, direction_description)
        """
        min_alt = self.MIN_ALTITUDE
        max_alt = self.MAX_ALTITUDE

        # 统计导弹在上方和下方的数量和威胁
        missiles_above = 0
        missiles_below = 0
        threat_above = 0  # 上方导弹的威胁度（距离越近越高）
        threat_below = 0

        for m in missiles:
            if m['altitude'] > current_alt:
                missiles_above += 1
                threat_above += 1.0 / m['distance']
            else:
                missiles_below += 1
                threat_below += 1.0 / m['distance']

        # 计算上升和下降的可用空间
        space_up = max_alt - current_alt
        space_down = current_alt - min_alt

        # 决策逻辑：
        # 1. 如果一侧导弹明显更多/威胁更大，向另一侧躲避
        # 2. 如果两侧相近，选择空间更大的方向
        # 3. 如果接近边界，利用边界进行极限规避

        # 计算上升和下降的得分
        score_up = 0
        score_down = 0

        # 导弹分布得分（远离导弹密集区域）
        if missiles_above > missiles_below:
            score_down += 30
        elif missiles_below > missiles_above:
            score_up += 30

        # 威胁度得分（远离威胁更大的方向）
        if threat_above > threat_below * 1.2:
            score_down += 40
        elif threat_below > threat_above * 1.2:
            score_up += 40

        # 空间得分（选择空间更大的方向）
        score_up += (space_up / 5000) * 20  # 最多20分
        score_down += (space_down / 5000) * 20

        # 加权平均高度得分（远离导弹平均高度）
        if weighted_avg_missile_alt > current_alt:
            score_down += 25
        else:
            score_up += 25

        # 决定方向
        if score_up >= score_down:
            target_alt = current_alt + alt_change
            direction = "上升"
        else:
            target_alt = current_alt - alt_change
            direction = "下降"

        # 应用高度限制
        target_alt = max(min_alt, min(max_alt, target_alt))

        # 极限规避：如果接近边界，直接到达边界
        if target_alt <= min_alt + 300:
            target_alt = min_alt
            direction = "极限俯冲"
        elif target_alt >= max_alt - 300:
            target_alt = max_alt
            direction = "极限爬升"

        return target_alt, direction

    def _calculate_composite_horizontal_offset(self, u_lon, u_lat, threat_info, agent):
        """
        计算综合水平偏移（配合垂直机动）

        策略：
        1. 选择垂直于导弹加权平均来袭方向的偏移
        2. 选择离边界更远、更安全的方向
        3. 保持小幅度偏移，主要依靠垂直躲避
        """
        # 评估左右偏移方向
        left_dir = threat_info['evade_perpendicular_left']
        right_dir = threat_info['evade_perpendicular_right']

        left_score = self._evaluate_direction(u_lon, u_lat, left_dir, agent)
        right_score = self._evaluate_direction(u_lon, u_lat, right_dir, agent)

        # 选择得分更高的方向
        offset_direction = left_dir if left_score >= right_score else right_dir

        # 计算偏移后的位置
        lon_off, lat_off = YxGeoUtils.km_to_lon_lat(
            u_lat, self.HORIZONTAL_EVADE_DISTANCE_KM, offset_direction
        )
        evade_lon = u_lon + lon_off
        evade_lat = u_lat + lat_off

        # 边界约束
        bf = agent.battlefield
        if bf.get('min_lon') is not None:
            margin = self.BOUNDARY_MARGIN
            evade_lon = max(bf['min_lon'] + margin, min(bf['max_lon'] - margin, evade_lon))
            evade_lat = max(bf['min_lat'] + margin, min(bf['max_lat'] - margin, evade_lat))

        return evade_lon, evade_lat
