"""
QVGA智能体接口

将训练好的个体封装为比赛用Agent
"""
import os
import sys
import numpy as np
import torch
from typing import List, Dict, Optional

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from env.agent.agent_base import AutoAgentBase
from utilities.yxScriptTreeFunc import YxScriptTreeFunc as decCmd
from utilities.yxGeoUtils import YxGeoUtils

from .individual import Individual, PolicyNetwork, decode_weights


class QVGAAutoAgent(AutoAgentBase):
    """
    QVGA智能体 - 用于比赛的Agent接口

    加载训练好的个体权重，执行推理
    """

    # 状态维度
    NUM_OWN_UNITS = 10
    NUM_ENEMY_UNITS = 10
    OWN_UNIT_FEATURES = 12
    ENEMY_UNIT_FEATURES = 10
    GLOBAL_FEATURES = 10
    STATE_DIM = NUM_OWN_UNITS * OWN_UNIT_FEATURES + NUM_ENEMY_UNITS * ENEMY_UNIT_FEATURES + GLOBAL_FEATURES

    # 动作解码参数
    DIRECTION_ANGLES = [90, 45, 0, -45, -90, -135, 180, 135]  # 8方向
    DISTANCES = [5, 15, 30]  # km
    SPEEDS = [200, 350, 500]  # m/s

    def __init__(self, side: str, name: str,
                 model_path: Optional[str] = None,
                 individual: Optional[Individual] = None,
                 device: str = 'auto'):
        """
        Args:
            side: 'red' 或 'blue'
            name: 智能体名称
            model_path: 模型路径 (.npz文件)
            individual: 直接传入Individual对象
            device: 计算设备
        """
        super().__init__(side, name)

        # 设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 策略网络
        self.policy_network = PolicyNetwork(
            state_dim=self.STATE_DIM,
            action_dim=40  # 10单位 × 4动作
        ).to(self.device)
        self.policy_network.eval()

        # 加载权重
        if individual is not None:
            decode_weights(individual.weights, self.policy_network)
            print(f"Loaded individual with fitness {individual.fitness:.2f}")
        elif model_path is not None and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print("Warning: Using random policy network")

        # 战场中心
        self.battlefield_center = (
            (self.battlefield['min_lat'] + self.battlefield['max_lat']) / 2,
            (self.battlefield['min_lon'] + self.battlefield['max_lon']) / 2
        )

    def _load_model(self, path: str):
        """加载模型"""
        data = np.load(path)
        weights = data['weights']
        decode_weights(weights, self.policy_network)
        print(f"Model loaded from {path}")
        if 'fitness' in data:
            print(f"  Fitness: {data['fitness']:.2f}")

    def update_decision(self, new_observation: Dict) -> List[Dict]:
        """
        主决策函数

        Args:
            new_observation: 仿真态势数据

        Returns:
            动作指令列表
        """
        self.last_observation = self.observation
        self.observation = new_observation

        # 1. 提取状态特征
        state = self._extract_state(new_observation)

        # 2. 策略网络推理
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_logits = self.policy_network(state_t)  # (1, 40)
            action_logits = action_logits.cpu().numpy()[0]

        # 3. 解码为指令
        commands = self._decode_actions(action_logits, new_observation)

        return commands

    def _extract_state(self, observation: Dict) -> np.ndarray:
        """提取状态特征"""
        features = []

        # 战场中心
        center_lat = self.battlefield_center[0]
        center_lon = self.battlefield_center[1]

        # 1. 己方单位特征
        own_units = observation.get('platform_list', [])
        for i in range(self.NUM_OWN_UNITS):
            if i < len(own_units):
                unit = own_units[i]
                feat = self._extract_unit_features(unit, center_lat, center_lon)
            else:
                feat = np.zeros(self.OWN_UNIT_FEATURES)
            features.extend(feat)

        # 2. 敌方单位特征
        enemy_tracks = [t for t in observation.get('track_list', [])
                        if t.get('platform_entity_type') != '导弹']
        for i in range(self.NUM_ENEMY_UNITS):
            if i < len(enemy_tracks):
                track = enemy_tracks[i]
                feat = self._extract_track_features(track, center_lat, center_lon)
            else:
                feat = np.zeros(self.ENEMY_UNIT_FEATURES)
            features.extend(feat)

        # 3. 全局特征
        global_feat = self._extract_global_features(observation, own_units, enemy_tracks)
        features.extend(global_feat)

        return np.array(features, dtype=np.float32)

    def _extract_unit_features(self, unit: Dict, center_lat: float, center_lon: float) -> List[float]:
        """提取己方单位特征"""
        lat = (unit.get('latitude', 0) - center_lat) / 1.0
        lon = (unit.get('longitude', 0) - center_lon) / 1.0
        alt = unit.get('altitude', 3000) / 10000.0
        speed = unit.get('speed', 0) / 500.0
        vx = unit.get('vx', 0) / 500.0
        vy = unit.get('vy', 0) / 500.0
        vz = unit.get('vz', 0) / 100.0
        heading = unit.get('heading', 0) / np.pi
        ammo = sum(w.get('quantity', 0) for w in unit.get('weapons', [])) / 10.0
        unit_type = 1.0 if unit.get('type') == '有人机' else 0.0
        alive = 1.0
        fuel = unit.get('fuel', 1.0)

        return [lat, lon, alt, speed, vx, vy, vz, heading, ammo, unit_type, alive, fuel]

    def _extract_track_features(self, track: Dict, center_lat: float, center_lon: float) -> List[float]:
        """提取敌方单位特征"""
        lat = (track.get('latitude', 0) - center_lat) / 1.0
        lon = (track.get('longitude', 0) - center_lon) / 1.0
        alt = track.get('altitude', 3000) / 10000.0
        speed = track.get('speed', 0) / 500.0
        vx = track.get('vx', 0) / 500.0
        vy = track.get('vy', 0) / 500.0
        heading = track.get('heading', 0) / np.pi
        fired_num = track.get('is_fired_num', 0) / 4.0
        visible = 1.0
        unit_type = 1.0 if track.get('platform_entity_type') == '有人机' else 0.0

        return [lat, lon, alt, speed, vx, vy, heading, fired_num, visible, unit_type]

    def _extract_global_features(self, obs: Dict, own_units: List, enemy_tracks: List) -> List[float]:
        """提取全局特征"""
        own_manned = sum(1 for u in own_units if u.get('type') == '有人机') / 2.0
        own_uav = sum(1 for u in own_units if u.get('type') == '无人机') / 8.0
        enemy_visible = len(enemy_tracks) / 10.0

        missiles = [t for t in obs.get('track_list', [])
                    if t.get('platform_entity_type') == '导弹']
        missile_count = len(missiles) / 20.0

        sim_time = obs.get('sim_time', 0) / 600.0

        return [own_manned, own_uav, enemy_visible, missile_count, sim_time,
                0.0, 0.0, 0.0, 0.0, 0.0]

    def _decode_actions(self, action_logits: np.ndarray, observation: Dict) -> List[Dict]:
        """
        解码动作logits为仿真指令

        action_logits: shape=(40,), 每4个元素对应一个单位
        """
        commands = []
        own_units = observation.get('platform_list', [])
        enemy_tracks = [t for t in observation.get('track_list', [])
                        if t.get('platform_entity_type') != '导弹']

        for i, unit in enumerate(own_units):
            if i >= self.NUM_OWN_UNITS:
                break

            # 提取该单位的动作logits
            unit_logits = action_logits[i * 4: (i + 1) * 4]

            # 解码各维度
            # 简化: 将连续值映射到离散动作
            direction_idx = int(np.clip(unit_logits[0] * 4 + 4, 0, 7))
            distance_idx = int(np.clip(unit_logits[1] * 1.5 + 1.5, 0, 2))
            speed_idx = int(np.clip(unit_logits[2] * 1.5 + 1.5, 0, 2))
            fire_score = unit_logits[3]

            unit_name = unit['name']
            unit_lat = unit.get('latitude', 0)
            unit_lon = unit.get('longitude', 0)
            unit_alt = unit.get('altitude', 3000)

            # 计算目标点
            angle_deg = self.DIRECTION_ANGLES[direction_idx]
            dist_km = self.DISTANCES[distance_idx]
            speed = self.SPEEDS[speed_idx]

            lon_offset, lat_offset = YxGeoUtils.km_to_lon_lat(
                self.battlefield_center[0], dist_km, angle_deg
            )

            target_lat = unit_lat + lat_offset
            target_lon = unit_lon + lon_offset
            target_alt = unit_alt

            # 生成fly_to_point指令
            commands.append(
                decCmd.fly_to_point(unit_name, (target_lat, target_lon, target_alt), speed)
            )

            # 开火指令
            if fire_score > 0.5 and len(enemy_tracks) > 0:
                # 找最近的敌人
                closest_enemy = None
                min_dist = float('inf')
                for enemy in enemy_tracks:
                    dist = YxGeoUtils.haversine_distance(
                        unit_lon, unit_lat,
                        enemy.get('longitude', 0), enemy.get('latitude', 0)
                    )
                    if dist < min_dist:
                        min_dist = dist
                        closest_enemy = enemy

                if closest_enemy:
                    # 检查射程
                    weapon_range = 20000 if unit.get('type') == '有人机' else 15000
                    has_ammo = any(w.get('quantity', 0) > 0 for w in unit.get('weapons', []))

                    if min_dist <= weapon_range and has_ammo:
                        commands.append(
                            decCmd.fire_track(unit_name, closest_enemy.get('target_name'))
                        )

        # 过滤重复指令
        commands = self._filter_commands(commands)

        return commands

    def _filter_commands(self, commands: List[Dict]) -> List[Dict]:
        """过滤重复指令"""
        seen_units = set()
        filtered = []

        for cmd in commands:
            if cmd is None:
                continue

            if cmd['func_name'] == 'fly_to_point':
                unit_name = cmd['args'][0]
                if unit_name not in seen_units:
                    filtered.append(cmd)
                    seen_units.add(unit_name)
            else:
                filtered.append(cmd)

        return filtered


# 测试代码
if __name__ == "__main__":
    # 创建智能体（使用随机权重）
    agent = QVGAAutoAgent('red', 'test_qvga', device='cpu')
    print(f"Agent created: {agent.name}")

    # 模拟观测
    mock_obs = {
        'side': 'red',
        'platform_list': [
            {'name': '红有人机1', 'type': '有人机', 'latitude': 21.0, 'longitude': 110.5,
             'altitude': 4000, 'speed': 300, 'heading': 0, 'weapons': [{'quantity': 4}]},
        ],
        'track_list': [
            {'target_name': '蓝无人机1', 'platform_entity_type': '无人机',
             'platform_entity_side': 'blue', 'latitude': 21.0, 'longitude': 111.5,
             'altitude': 3500, 'is_fired_num': 0}
        ]
    }

    # 测试决策
    commands = agent.update_decision(mock_obs)
    print(f"Generated {len(commands)} commands:")
    for cmd in commands:
        print(f"  {cmd['func_name']}: {cmd['args'][0]}")
