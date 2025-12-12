"""
主智能体类 (BTDemoAgent)

基于行为树的战斗AI智能体
"""

import math
import random
from typing import Dict

from env.agent.agent_base import AutoAgentBase
from utilities.yxGeoUtils import YxGeoUtils

from .bt_framework import Sequence, Selector
from .actions import (
    ActionResetFrame,
    ConditionCheckInitialDeployment,
    ActionExecuteDeployment,
    ActionEvadeMissilesAdvanced,
    ActionProtectMannedVision,
    ActionMannedRetreat,
    ActionAttackLogic,
    ActionSearchFormation,
    ActionCenterPatrol,
    ActionPatrolFormation
)


class BTDemoAgent(AutoAgentBase):
    """
    基于行为树的战斗AI智能体

    行为树结构：
    1. 优先检查是否需要开局部署
    2. 进入战斗循环：
       a. 重置帧数据
       b. 导弹规避（最高优先级）
       c. 无弹药无人机保护有人机
       d. 有人机后撤
       e. 攻击逻辑
       f. 搜索阵型
       g. 中心巡逻
       h. 防御巡逻
    """

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
        self.bt_root = Selector([
            # 分支 1: 开局部署
            Sequence([
                ConditionCheckInitialDeployment(),
                ActionExecuteDeployment()
            ]),

            # 分支 2: 常规战斗循环 (Main Loop)
            Sequence([
                ActionResetFrame(),             # 步骤1: 清理
                ActionEvadeMissilesAdvanced(),  # 步骤2: 导弹规避（高级版，有人机优先保护）
                ActionProtectMannedVision(),    # 步骤3: 无弹药无人机→保护有人机视野
                # ActionMannedRetreat(),          # 步骤4: 发现敌机→有人机后撤
                ActionAttackLogic(),            # 步骤5: 开火逻辑
                ActionSearchFormation(),        # 步骤6: 无敌机→分散搜索推进
                ActionCenterPatrol(),           # 步骤7: 到达中心无敌机→盘旋
                ActionPatrolFormation()         # 步骤8: 兜底
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
        """预测威胁"""
        threatened = []
        m_lon, m_lat = missile['longitude'], missile['latitude']
        m_heading = (math.degrees(missile.get('heading', 0)) + 360) % 360

        for unit in self.own_units:
            dist = YxGeoUtils.haversine_distance(unit['longitude'], unit['latitude'], m_lon, m_lat)
            bearing = YxGeoUtils.calculate_bearing(m_lon, m_lat, unit['longitude'], unit['latitude'])
            angle_diff = abs((bearing - m_heading + 180) % 360 - 180)  # 归一化角度差

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

    def save_battle_data(self):
        """
        保存战斗数据（击杀/脱靶记录）

        在战斗结束后调用此方法保存收集的数据
        用于后续分析和拟合致死区间模型
        """
        ActionAttackLogic.save_kill_data(self)

    def print_battle_summary(self):
        """打印战斗数据统计摘要"""
        if hasattr(self, 'kill_data_records') and self.kill_data_records:
            ActionAttackLogic.print_kill_data_summary(self.kill_data_records)
        else:
            print("[战斗摘要] 没有收集到导弹数据")
