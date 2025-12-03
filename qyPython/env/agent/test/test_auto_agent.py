import math
import random
from typing import List, Dict

from env.agent.agent_base import AutoAgentBase
from utilities.yxScriptTreeFunc import YxScriptTreeFunc as decCmd
from utilities.yxGeoUtils import YxGeoUtils


class TestAutoAgent(AutoAgentBase):
    """
    测试用智能体 - 用于跑通流程
    基于demo_auto_agent简化而来
    """
    def __init__(self, side, name):
        super().__init__(side, name)

        # 状态变量
        self.initial_deployment_complete = False

        # 态势信息
        self.own_units = []      # 己方单位
        self.enemy_units = []    # 敌方单位
        self.enemy_missiles = [] # 敌方导弹

        # 战场中心点
        self.battlefield_center_lat = (self.battlefield['min_lat'] + self.battlefield['max_lat']) / 2
        self.battlefield_center_lon = (self.battlefield['min_lon'] + self.battlefield['max_lon']) / 2

        # 帧计数
        self.frame_count = 0

    def update_decision(self, new_observation: Dict) -> List[Dict]:
        """
        主决策函数
        """
        self.last_observation = self.observation
        self.observation = new_observation
        self.frame_count += 1

        # 1. 解析态势
        self._parse_observation(new_observation)

        if not self.own_units:
            return []

        # 2. 初始部署
        if not self.initial_deployment_complete:
            actions = self._execute_initial_deployment()
            self.initial_deployment_complete = True
            return actions

        actions = []

        # 3. 导弹规避
        evade_actions = self._evade_missiles()
        actions.extend(evade_actions)

        # 4. 战术执行
        if self.enemy_units:
            # 有敌人则攻击
            attack_actions = self._execute_attack()
            actions.extend(attack_actions)
        else:
            # 无敌人则巡逻
            patrol_actions = self._execute_patrol()
            actions.extend(patrol_actions)

        # 5. 去重（同一平台只保留第一个fly_to_point）
        actions = self._filter_actions(actions)

        return actions

    def _parse_observation(self, observation: Dict):
        """解析态势数据"""
        self.own_units = observation.get('platform_list', [])

        self.enemy_units = []
        self.enemy_missiles = []

        for track in observation.get('track_list', []):
            if track.get('platform_entity_side') != self.side:
                if track.get('platform_entity_type') == '导弹':
                    self.enemy_missiles.append(track)
                else:
                    self.enemy_units.append(track)

    def _execute_initial_deployment(self) -> List[Dict]:
        """初始部署：向战场中心前进"""
        actions = []

        for unit in self.own_units:
            unit_type = unit.get('type')

            # 根据side决定前进方向
            if self.side == 'red':
                lon_offset = 0.4  # 向东（向中心）
            else:
                lon_offset = -0.4  # 向西（向中心）

            if unit_type == '有人机':
                # 有人机靠后，速度慢
                target = (
                    unit.get('latitude'),
                    unit.get('longitude') + lon_offset * 0.7,
                    4000
                )
                actions.append(decCmd.fly_to_point(unit['name'], target, 300))
            else:
                # 无人机靠前，速度快
                target = (
                    unit.get('latitude'),
                    unit.get('longitude') + lon_offset,
                    3500
                )
                actions.append(decCmd.fly_to_point(unit['name'], target, 400))

        return actions

    def _evade_missiles(self) -> List[Dict]:
        """简单导弹规避"""
        actions = []

        for missile in self.enemy_missiles:
            missile_lat = missile.get('latitude')
            missile_lon = missile.get('longitude')

            # 找到距离导弹最近的己方单位
            for unit in self.own_units:
                distance = YxGeoUtils.haversine_distance(
                    unit.get('longitude'), unit.get('latitude'),
                    missile_lon, missile_lat
                )

                # 距离小于8km时规避
                if distance < 8000:
                    # 简单规避：向侧面移动
                    evade_lat = unit.get('latitude') + 0.03  # 向北移动约3km
                    evade_lon = unit.get('longitude')
                    evade_alt = unit.get('altitude', 3000) + 500

                    actions.append(decCmd.fly_to_point(
                        unit['name'],
                        (evade_lat, evade_lon, evade_alt),
                        700  # 高速规避
                    ))

        return actions

    def _execute_attack(self) -> List[Dict]:
        """攻击战术：向敌人靠近并开火"""
        actions = []

        # 为每个己方单位找最近的敌人
        for unit in self.own_units:
            closest_enemy = None
            min_distance = float('inf')

            for enemy in self.enemy_units:
                dist = YxGeoUtils.haversine_distance(
                    unit.get('longitude'), unit.get('latitude'),
                    enemy.get('longitude'), enemy.get('latitude')
                )
                if dist < min_distance:
                    min_distance = dist
                    closest_enemy = enemy

            if closest_enemy:
                # 武器射程
                weapon_range = 20000 if unit.get('type') == '有人机' else 15000

                # 在射程内且有弹药则开火
                if min_distance <= weapon_range * 0.9:
                    has_ammo = any(w.get('quantity', 0) > 0 for w in unit.get('weapons', []))
                    if has_ammo and closest_enemy.get('is_fired_num', 0) < 4:
                        actions.append(decCmd.fire_track(unit['name'], closest_enemy.get('target_name')))

                # 向敌人方向移动（保持在射程边缘）
                target = (
                    closest_enemy.get('latitude'),
                    closest_enemy.get('longitude'),
                    closest_enemy.get('altitude', 3000)
                )
                actions.append(decCmd.fly_to_point(unit['name'], target, 450))

        return actions

    def _execute_patrol(self) -> List[Dict]:
        """巡逻战术：在战场中心区域巡逻"""
        actions = []

        for i, unit in enumerate(self.own_units):
            # 计算巡逻点（围绕战场中心）
            angle = (i * 36 + self.frame_count) % 360
            radius = 20 if unit.get('type') == '有人机' else 35  # km

            lon_offset, lat_offset = YxGeoUtils.km_to_lon_lat(
                self.battlefield_center_lat, radius, angle
            )

            target = (
                self.battlefield_center_lat + lat_offset,
                self.battlefield_center_lon + lon_offset,
                4000 if unit.get('type') == '有人机' else 3500
            )

            actions.append(decCmd.fly_to_point(unit['name'], target, 300))

        return actions

    def _filter_actions(self, actions: List[Dict]) -> List[Dict]:
        """过滤重复指令"""
        seen_units = set()
        filtered = []

        for action in actions:
            if action is None:
                continue

            if action['func_name'] == 'fly_to_point':
                unit_name = action['args'][0]
                if unit_name not in seen_units:
                    filtered.append(action)
                    seen_units.add(unit_name)
            else:
                # 非fly_to_point指令直接保留
                filtered.append(action)

        return filtered
