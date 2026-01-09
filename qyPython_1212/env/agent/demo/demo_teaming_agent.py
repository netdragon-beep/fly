import config, random, math
from typing import List, Dict, Tuple
from env.agent.agent_base import TeamingAgentBase, teaming_task
from utilities.yxRegistry import YxRegistry
from utilities.yxScriptTreeFunc import YxScriptTreeFunc as callFunc
from utilities.yxGeoUtils import YxGeoUtils


class DemoTeamingAgent(TeamingAgentBase):
    def __init__(self, side, name):
        super().__init__(side, name)

        # 战术状态变
        self.initial_deployment_complete = False  # 初始部署是否完成
        # 全局态势信息
        self.own_units = []  # 己方单位
        self.enemy_units = []  # 敌方单位（已探测）
        self.enemy_missiles = []  # 敌方导弹（已探测）
        self.own_missiles = []  # 己方导弹
        self.targets_destroyed = set()  # 已摧毁目标ID集合

        # 战场中心点
        self.battlefield_center_lat = (self.battlefield['min_lat'] + self.battlefield['max_lat']) / 2
        self.battlefield_center_lon = (self.battlefield['min_lon'] + self.battlefield['max_lon']) / 2
        self.defense_angle_offset = random.randint(-45, 45)

        # 帧计数器
        self.frame_count = 0


    def update_decision(self, new_observation: dict):
        """更新决策逻辑"""
        self.last_observation = self.observation
        self.observation = new_observation
        self.frame_count += 1  # 增加帧计数

        # 1. 解析全局态势数据
        self._parse_observation(new_observation)

        # 2. 执行初始部署（仅在第一次更新时）
        if not self.initial_deployment_complete:
            initial_actions = self._execute_initial_deployment()
            self.initial_deployment_complete = True
            return initial_actions

        actions = []
        # 没有探测到敌机，执行防御阵型
        defense_actions = self._execute_defense_formation()
        actions.extend(defense_actions)

        return actions

    def _execute_defense_formation(self) -> List[Dict]:
        """执行防御阵型：所有飞机均匀分布在以战场中心为圆心的圆形区域，有人机构成内圆，无人机构成外圆"""
        actions = []

        uavs = [u for u in self.own_units if u.get('type') == '无人机']
        outer_radius = 30  # km
        for i, unit in enumerate(uavs):
            angle = ((i / len(uavs)) * 360 +self.defense_angle_offset+360)%360  # 均匀分布角度
            lon_offset, lat_offset = YxGeoUtils.km_to_lon_lat(self.battlefield_center_lat, outer_radius, angle)
            target_point = [
                self.battlefield_center_lat + lat_offset,
                self.battlefield_center_lon + lon_offset,
                3500  # 高度3500m
            ]
            actions.append(callFunc.fly_to_point(unit['name'], target_point, 300))

        return actions

    def _parse_observation(self, observation: dict):
        """解析观测数据，提取己方单位、敌方单位和导弹"""
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

        uavs = [u for u in self.own_units if u.get('type') == '无人机']

        # 无人机靠前，有人机靠后
        for i, unit in enumerate(uavs):
            # 无人机位置：向前50km，高度3500m
            target_point = [
                unit.get('latitude'),
                unit.get('longitude') + 0.45 if self.side == 'red' else unit.get('longitude') - 0.45,  # 向中心飞行
                3500
            ]
            actions.append(callFunc.fly_to_point(unit['name'], target_point, 400))
        return actions


@YxRegistry.register("单平台任务示例", 0)
@teaming_task
def task1(selected_forces: List[str], raw_observation: Dict):
    print(f"{selected_forces}执行单平台任务")
    actions = []
    # TODO: 单平台任务决策逻辑
    return actions

@YxRegistry.register("编组任务示例", 1)
@teaming_task
def task2(selected_forces: List[str], raw_observation: Dict):
    print(f"{selected_forces}执行编组任务")
    actions = []
    # TODO: 编组任务决策逻辑
    return actions

@YxRegistry.register("通用任务示例", 2)
@teaming_task
def task3(selected_forces: List[str], raw_observation: Dict):
    print(f"{selected_forces}执行通用任务")
    actions = []
    # TODO: 通用任务决策逻辑
    return actions