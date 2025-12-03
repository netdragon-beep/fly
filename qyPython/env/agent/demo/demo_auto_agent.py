import math
import random, config
from typing import List, Dict, Tuple

from env.agent.agent_base import AutoAgentBase
from utilities.yxScriptTreeFunc import YxScriptTreeFunc as decCmd
from utilities.yxGeoUtils import YxGeoUtils


class DemoAutoAgent(AutoAgentBase):
    def __init__(self, side, name):
        super().__init__(side, name)

        # 战术状态变量

        self.initial_deployment_complete = False  # 初始部署是否完成

        # 全局态势信息
        self.own_units = []  # 己方单位
        self.enemy_units = []  # 敌方单位（已探测）
        self.enemy_missiles = []  # 敌方导弹（已探测）
        self.own_missiles = []  # 己方导弹
        self.targets_destroyed = set()  # 已摧毁目标ID集合

        # 导弹发射记录
        self.missile_launch_records = {}  # {目标ID: {发射单位: 发射时间/帧数}}

        # 历史指令记录（用于跟踪单位目标点）
        self.unit_orders = {}  # {单位名称: 最新指令}

        # 战场中心点
        self.battlefield_center_lat = (self.battlefield['min_lat'] + self.battlefield['max_lat']) / 2
        self.battlefield_center_lon = (self.battlefield['min_lon'] + self.battlefield['max_lon']) / 2
        self.defense_angle_offset = random.randint(-45, 45)

        # 帧计数器
        self.frame_count = 0

    def update_decision(self, new_observation: Dict):
        """更新决策逻辑"""
        self.last_observation = self.observation
        self.observation = new_observation
        self.frame_count += 1  # 增加帧计数

        # 1. 解析全局态势数据
        self._parse_observation(new_observation)

        if not self.own_units:
            return []  # 没有己方单位数据，不采取行动

        # 2. 执行初始部署（仅在第一次更新时）
        if not self.initial_deployment_complete:
            initial_actions = self._execute_initial_deployment()
            self.initial_deployment_complete = True
            return initial_actions

        # 3. 优先处理防御动作：提前规避导弹
        actions = self._evade_missiles()

        # 4. 根据是否有敌机执行不同战术
        if self.enemy_units:
            # 探测到敌机，执行攻击战术
            attack_actions = self._execute_attack_tactics()
            actions.extend(attack_actions)
        else:
            # 没有探测到敌机，执行防御阵型
            defense_actions = self._execute_defense_formation()
            actions.extend(defense_actions)

        # 同一平台的多个指令，只对fly_to_point指令保留第一个指令
        action_units = []
        filtered_actions = []

        for action in actions:
            if action is not None:
                if action['func_name'] == 'fly_to_point':
                    # 对于fly_to_point指令，检查是否已经为该单位添加过指令
                    if action['args'][0] not in action_units:
                        filtered_actions.append(action)
                        action_units.append(action['args'][0])
                else:
                    # 对于非fly_to_point指令（如fire_track），直接保留
                    filtered_actions.append(action)

        actions = filtered_actions

        # 记录发出的指令
        for action in actions:
            unit_name = action['args'][0]
            self.unit_orders[unit_name] = action

        return actions

    def _parse_observation(self, observation: Dict):
        """解析态势数据，提取己方单位、敌方单位和导弹"""
        assert observation.get('side') == self.side, f"side must be {self.side}"

        # 提取己方平台
        self.own_units = observation.get('platform_list', [])

        # 提取敌方单位和导弹
        self.enemy_units = []
        self.enemy_missiles = []

        for track in observation.get('track_list', []):
            if track.get('platform_entity_side') != self.side:
                if track.get('platform_entity_type') == '导弹':
                    self.enemy_missiles.append(track)
                else:
                    self.enemy_units.append(track)

    def _execute_initial_deployment(self) -> List[Dict]:
        """执行初始部署：所有飞机向中心飞，有人机靠后，无人机靠前"""
        actions = []

        # 分离有人机和无人机
        manned_aircraft = [u for u in self.own_units if u.get('type') == '有人机']
        uavs = [u for u in self.own_units if u.get('type') == '无人机']

        # 无人机靠前，有人机靠后
        for i, unit in enumerate(uavs):
            # 无人机位置：向前50km，高度3500m
            target_point = (
                unit.get('latitude'),
                unit.get('longitude') + 0.45 if self.side == 'red' else unit.get('longitude') - 0.45,  # 向中心飞行
                3500
            )
            actions.append(decCmd.fly_to_point(unit['name'], target_point, 400))

        for i, unit in enumerate(manned_aircraft):
            # 有人机位置：向前30km，高度4000m
            target_point = (
                unit.get('latitude'),
                unit.get('longitude') + 0.27 if self.side == 'red' else unit.get('longitude') - 0.27,  # 向中心飞行
                4000
            )
            actions.append(decCmd.fly_to_point(unit['name'], target_point, 300))

        return actions

    def _evade_missiles(self) -> List[Dict]:
        """利用全局信息提前规避导弹"""
        actions = []
        missile_avoided = set()

        for missile in self.enemy_missiles:
            missile_id = missile.get('target_id') or id(missile)
            if missile_id in missile_avoided:
                continue

            threatened_units = self._predict_threatened_units(missile)

            for unit in threatened_units:
                evade_direction = self._calculate_evade_direction(unit, missile)
                # 计算5km规避点（坐标顺序：纬度, 经度, 高度）
                lon_offset, lat_offset = YxGeoUtils.km_to_lon_lat(self.battlefield_center_lat, 5, evade_direction)
                evade_point = (
                    unit.get('latitude', 0) + lat_offset,  # 纬度（第一位）
                    unit.get('longitude', 0) + lon_offset,  # 经度（第二位）
                    unit.get('altitude', 3000) + 500  # 高度（第三位）
                )
                actions.append(decCmd.fly_to_point(unit['name'], evade_point, 700))

            missile_avoided.add(missile_id)

        return actions

    def _execute_attack_tactics(self) -> List[Dict]:
        """执行攻击战术：计算所有距离并排序，按距离发射导弹，然后所有单位向最近敌机靠近"""
        actions = []
        cur_round_missile_launch_records = {}

        # 第一步：计算所有我方单位与所有敌方单位的距离
        distance_pairs = []
        for unit in self.own_units:
            for enemy in self.enemy_units:
                distance = YxGeoUtils.haversine_distance(
                    unit.get('longitude'), unit.get('latitude'),
                    enemy.get('longitude'), enemy.get('latitude')
                )
                distance_pairs.append({
                    'unit': unit,
                    'enemy': enemy,
                    'distance': distance
                })

        # 第二步：按照距离排序（从近到远）
        distance_pairs.sort(key=lambda x: x['distance'])

        # 第三步：按照距离排序发射导弹
        for pair in distance_pairs:
            unit = pair['unit']
            enemy = pair['enemy']
            distance = pair['distance']

            # 检查敌机是否可攻击（未被过度攻击）
            if enemy['is_fired_num'] + len(cur_round_missile_launch_records.get(enemy['target_id'], [])) >= 4:
                continue

            # 计算攻击距离边缘
            weapon_range = 20000 if unit.get('type') == '有人机' else 15000
            optimal_distance = weapon_range * 0.9  # 保持在最大射程的90%处

            # 如果在攻击范围内且我方有导弹，发射导弹
            if distance <= optimal_distance:
                if any(w.get('quantity', 0) > 0 for w in unit.get('weapons', [])):
                    # 更新武器数量
                    for weapon in unit.get('weapons', []):
                        if weapon.get('quantity', 0) > 0:
                            weapon['quantity'] -= 1
                            break
                    actions.append(decCmd.fire_track(unit['name'], enemy.get('target_name')))
                    if enemy['target_id'] not in cur_round_missile_launch_records:
                        cur_round_missile_launch_records[enemy['target_id']] = []
                    cur_round_missile_launch_records[enemy['target_id']].append(unit['name'])

        # 第四步：遍历我方所有单位，每个单位向最近敌方单位靠近并保持距离
        for unit in self.own_units:
            # 为每个我方单位找到最近的敌机
            closest_enemy = None
            min_distance = float('inf')

            for enemy in self.enemy_units:
                distance = YxGeoUtils.haversine_distance(
                    unit.get('longitude'), unit.get('latitude'),
                    enemy.get('longitude'), enemy.get('latitude')
                )

                if distance < min_distance:
                    min_distance = distance
                    closest_enemy = enemy

            if closest_enemy:
                # 计算攻击距离边缘
                weapon_range = 20000 if unit.get('type') == '有人机' else 15000
                optimal_distance = weapon_range * 0.9

                # 向目标敌机机动并保持距离
                direction = YxGeoUtils.calculate_direction_to(
                    closest_enemy.get('longitude'), closest_enemy.get('latitude'),
                    unit.get('longitude'), unit.get('latitude')
                )

                lon_offset, lat_offset = YxGeoUtils.km_to_lon_lat(self.battlefield_center_lat, optimal_distance / 1000,
                                                                  direction)
                target_point = (
                    closest_enemy.get('latitude') + lat_offset,
                    closest_enemy.get('longitude') + lon_offset,
                    closest_enemy.get('altitude', 3000)
                )
                actions.append(decCmd.fly_to_point(unit['name'], target_point, 500))

        return actions

    def _execute_defense_formation(self) -> List[Dict]:
        """执行防御阵型：所有飞机均匀分布在以战场中心为圆心的圆形区域，有人机构成内圆，无人机构成外圆"""
        actions = []

        # 分离有人机和无人机
        manned_aircraft = [u for u in self.own_units if u.get('type') == '有人机']
        uavs = [u for u in self.own_units if u.get('type') == '无人机']

        # 有人机构成内圆（半径15km）
        inner_radius = 15  # km
        for i, unit in enumerate(manned_aircraft):
            angle = ((i / len(manned_aircraft)) * 360 + self.defense_angle_offset + 360) % 360  # 均匀分布角度
            lon_offset, lat_offset = YxGeoUtils.km_to_lon_lat(self.battlefield_center_lat, inner_radius, angle)
            target_point = (
                self.battlefield_center_lat + lat_offset,
                self.battlefield_center_lon + lon_offset,
                4000  # 高度4000m
            )
            actions.append(decCmd.fly_to_point(unit['name'], target_point, 300))

        # 无人机构成外圆（半径30km）
        outer_radius = 30  # km
        for i, unit in enumerate(uavs):
            angle = ((i / len(uavs)) * 360 + self.defense_angle_offset + 360) % 360  # 均匀分布角度
            lon_offset, lat_offset = YxGeoUtils.km_to_lon_lat(self.battlefield_center_lat, outer_radius, angle)
            target_point = (
                self.battlefield_center_lat + lat_offset,
                self.battlefield_center_lon + lon_offset,
                3500  # 高度3500m
            )
            actions.append(decCmd.fly_to_point(unit['name'], target_point, 300))

        return actions

    def has_launched_recently(self, unit_name: str, target_id: str) -> bool:
        """检查单位是否最近对目标发射过导弹（避免重复发射）"""
        if target_id not in self.missile_launch_records:
            return False

        if unit_name in self.missile_launch_records[target_id]:
            # 检查发射时间（帧数），如果是在最近10帧内发射的，则认为已经发射过
            launch_frame = self.missile_launch_records[target_id][unit_name]
            if self.frame_count - launch_frame < 10:  # 10帧内不再发射
                return True

        return False

    def record_missile_launch(self, unit_name: str, target_id: str):
        """记录导弹发射"""
        if target_id not in self.missile_launch_records:
            self.missile_launch_records[target_id] = {}

        self.missile_launch_records[target_id][unit_name] = self.frame_count

    def _predict_threatened_units(self, missile: Dict) -> List[Dict]:
        """预测受导弹威胁的单位（简化实现）"""
        threatened = []
        missile_lon = missile.get('longitude')
        missile_lat = missile.get('latitude')
        missile_heading = missile.get('heading', 0)  # 导弹航向（弧度制，-pi到pi）

        for unit in self.own_units:
            # 计算单位与导弹的距离
            distance = YxGeoUtils.haversine_distance(
                unit.get('longitude'), unit.get('latitude'),
                missile_lon, missile_lat
            )

            # 计算单位是否在导弹航向的15度范围内
            unit_bearing = YxGeoUtils.calculate_bearing(missile_lon, missile_lat,
                                                        unit.get('longitude'), unit.get('latitude'))

            # 将导弹航向从弧度转换为角度，并确保在0-360度范围内
            missile_heading_deg = math.degrees(missile_heading)  # 转换为角度
            missile_heading_deg = (missile_heading_deg + 360) % 360  # 确保在0-360度范围内

            angle_diff = abs((unit_bearing - missile_heading_deg) % 360)
            # 确保取最小角度差（考虑360度循环）
            angle_diff = min(angle_diff, 360 - angle_diff)

            # 距离10km内且在航向15度范围内视为受威胁
            if distance < 10000 and angle_diff < 15:
                threatened.append(unit)

        return threatened

    def _calculate_evade_direction(self, unit: Dict, missile: Dict) -> float:
        """计算规避方向（与导弹来袭方向相反）"""
        unit_lon = unit.get('longitude')
        unit_lat = unit.get('latitude')
        missile_lon = missile.get('longitude')
        missile_lat = missile.get('latitude')

        # 计算导弹相对单位的方向（来袭方向）
        incoming_bearing = YxGeoUtils.calculate_bearing(
            missile_lon, missile_lat, unit_lon, unit_lat
        )

        # 规避方向：来袭方向加90度或减90度（侧向规避）
        # 选择远离战场边缘的方向
        battlefield_center_lat = (self.battlefield['min_lat'] + self.battlefield['max_lat']) / 2
        unit_lat = unit.get('latitude', battlefield_center_lat)

        if unit_lat < battlefield_center_lat:
            # 单位在战场下半部分，向上规避
            evade_angle = (incoming_bearing + 90) % 360
        else:
            # 单位在战场上半部分，向下规避
            evade_angle = (incoming_bearing - 90) % 360

        return evade_angle