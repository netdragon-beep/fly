# -*- coding: utf-8 -*-
"""
VEB-RL 

M: Gym ( VEB-RL 
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# 项目路径设置
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

try:
    import config
    from env.env import Env
    from env.agent.demo.demo_auto_agent import DemoAutoAgent
    HAS_SIM_ENV = True
except ImportError:
    HAS_SIM_ENV = False
    print("Warning: Simulation environment not available, using mock mode only")

# 支持直接运行和作为模块导入
try:
    from .reward import RewardCalculator, RewardConfig
except ImportError:
    from reward import RewardCalculator, RewardConfig


class VEBBattleEnvironment:
    """
    VEB-RL 

    Л Gym  (reset, step) ( VEB-RL 

     RealBattleEnvironment :+:
    - Л step/reset ^ battle 
    - /e
    - M VEB-RL  Q-learning A
    """

    def __init__(
        self,
        reward_config: Optional[RewardConfig] = None,
        state_dim: int = 230,
        action_dim: int = 60,
        device: str = 'cpu',
        opponent_agent=None,
        max_episode_steps: int = 1000
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.max_episode_steps = max_episode_steps

        # Vh
        self.reward_calculator = RewardCalculator(reward_config or RewardConfig())

        # 动作拆分：默认每个平台 8 个机动动作 + 4 个开火动作
        self.max_controlled_units = 5
        self.move_actions_per_unit = 8
        self.fire_actions_per_unit = 4
        self.actions_per_unit = self.move_actions_per_unit + self.fire_actions_per_unit
        expected_actions = self.max_controlled_units * self.actions_per_unit
        if self.action_dim != expected_actions:
            print(
                f"[VEBBattleEnv] action_dim={self.action_dim} 与期望 {expected_actions} 不一致，"
                f"仍按 {self.actions_per_unit} 动作/平台 解码。"
            )

        # (	
        self.has_sim = HAS_SIM_ENV
        if self.has_sim:
            self.env = Env(red_agent=None, blue_agent=None)
            self.opponent_agent = opponent_agent or DemoAutoAgent('blue', 'blue_demo')
        else:
            self.env = None
            self.opponent_agent = None

        # 
        self.current_state = None
        self.prev_obs = None
        self.step_count = 0
        self.episode_done = False
        self.episode_started = False

    def reset(self) -> np.ndarray:
        """
        n

        Returns:
            state: ˶y
        """
        self.step_count = 0
        self.episode_done = False
        self.prev_obs = None
        self.reward_calculator.reset()

        if self.has_sim and self.env is not None:
            # n
            self._reset_simulation()
            initial_obs = self._get_observation()
            self.current_state = self._extract_state(initial_obs)
            self.prev_obs = initial_obs
            self.reward_calculator.reset(initial_obs)
        else:
            # !!
            self.current_state = np.random.randn(self.state_dim).astype(np.float32) * 0.1

        self.episode_started = True
        return self.current_state

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        """
        执行动作 - 支持单智能体和多智能体模式

        Args:
            action: 动作，支持两种格式：
                - int: 单个动作索引 (兼容旧模式)
                - array-like: 多智能体动作 [action_0, action_1, ..., action_n]
                              每个元素是对应平台的动作 (0 ~ actions_per_unit-1)

        Returns:
            next_state, reward, done, info
        """
        if not self.episode_started:
            raise RuntimeError("Must call reset() before step()")

        self.step_count += 1

        # 转换动作格式
        if isinstance(action, (int, np.integer)):
            # 旧模式：单个动作，转换为多智能体动作
            multi_actions = self._single_to_multi_action(action)
        else:
            # 新模式：多智能体动作数组
            multi_actions = np.array(action)

        if self.has_sim and self.env is not None:
            next_state, reward, done, info = self._sim_step_multi(multi_actions)
        else:
            next_state, reward, done, info = self._mock_step_multi(multi_actions)

        self.current_state = next_state

        if self.step_count >= self.max_episode_steps:
            done = True
            info['truncated'] = True

        self.episode_done = done

        return next_state, reward, done, info

    def _single_to_multi_action(self, action: int) -> np.ndarray:
        """将单个动作转换为多智能体动作（兼容旧接口）"""
        unit_idx = action // self.actions_per_unit
        action_idx = action % self.actions_per_unit
        multi_actions = np.zeros(self.max_controlled_units, dtype=np.int32)
        if unit_idx < self.max_controlled_units:
            multi_actions[unit_idx] = action_idx
        return multi_actions


    def _sim_step_multi(self, multi_actions: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """多智能体仿真执行"""
        fun = self.env.funTool
        info = {}

        # 为每个平台生成并发送指令
        cmds = self._decode_multi_actions(multi_actions)
        self._send_cmds(cmds)

        # 执行对手动作
        raw_obs = copy.deepcopy(fun.get_sim_data())
        blue_side = self._get_side(raw_obs, 'blue')
        if self.opponent_agent is not None:
            blue_cmds = self.opponent_agent.get_cmds(blue_side)
            self._send_cmds(blue_cmds)

        # 推进仿真
        fun.sim_data.frame = config.frame_num if hasattr(config, 'frame_num') else 10
        fun.sim_control('step')
        time.sleep(0.05)

        # 获取新状态
        new_obs = self._get_observation()

        # 计算奖励
        if self.prev_obs is not None:
            reward = self.reward_calculator.step(self.prev_obs, new_obs, {})
        else:
            reward = 0.0

        next_state = self._extract_state(new_obs)

        # 检查是否结束
        done_flags = self.env.get_done()
        done = done_flags[0]

        if done:
            victory = bool(done_flags[1])
            self.reward_calculator.finalize(victory)
            info['victory'] = victory
            info['summary'] = self.reward_calculator.get_summary()

        self.prev_obs = new_obs
        return next_state, reward, done, info

    def _mock_step_multi(self, multi_actions: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """多智能体 Mock 执行"""
        # 根据每个平台的动作计算状态变化
        action_effect = np.zeros(self.state_dim, dtype=np.float32)
        
        for i, action in enumerate(multi_actions[:self.max_controlled_units]):
            offset = i * 10
            if action < self.move_actions_per_unit:
                # 机动动作：影响位置状态
                action_effect[offset:offset+3] = np.random.randn(3) * 0.1
            else:
                # 开火动作：影响其他状态
                action_effect[offset+5:offset+8] = np.random.randn(3) * 0.05

        next_state = self.current_state + action_effect
        next_state = np.clip(next_state, -10, 10).astype(np.float32)

        # 计算奖励（鼓励积极行动）
        reward = -np.sum(np.abs(next_state[:50])) * 0.01
        # 奖励开火动作
        fire_actions = sum(1 for a in multi_actions if a >= self.move_actions_per_unit)
        reward += fire_actions * 0.1

        done = np.random.random() < 0.01 or self.step_count >= self.max_episode_steps

        info = {}
        if done:
            info['victory'] = np.random.random() < 0.5

        return next_state, reward, done, info

    def _decode_multi_actions(self, multi_actions: np.ndarray) -> List[Dict]:
        """
        解码多智能体动作 - 每个平台独立执行自己的动作
        
        Args:
            multi_actions: shape=(num_agents,), 每个元素是该平台的动作索引 (0~11)
        
        Returns:
            cmds: 所有平台的指令列表
        """
        cmds: List[Dict] = []

        if not self.has_sim:
            return cmds

        raw_obs = copy.deepcopy(self.env.funTool.get_sim_data())
        red_side = self._get_side(raw_obs, 'red')
        platforms = red_side.get('platform_list', [])[:self.max_controlled_units]

        if not platforms:
            return cmds

        # 为每个平台生成指令
        for i, platform in enumerate(platforms):
            if i >= len(multi_actions):
                break
                
            action_idx = int(multi_actions[i])
            
            if action_idx < self.move_actions_per_unit:
                # 机动动作 (0-7)
                cmd = self._build_move_cmd(platform, action_idx)
            else:
                # 开火动作 (8-11)
                fire_slot = action_idx - self.move_actions_per_unit
                cmd = self._build_fire_cmd(platform, fire_slot, raw_obs)
                # 开火后也要继续机动
                if cmd:
                    cmds.append(cmd)
                # 开火的同时朝敌方移动
                enemy_center = self._get_enemy_center(raw_obs)
                cmd = self._build_move_toward_enemy(platform, enemy_center)
            
            if cmd:
                cmds.append(cmd)

        return cmds

    def _reset_simulation(self):
        """n"""
        if not self.has_sim:
            return

        fun = self.env.funTool

        # Iޥ
        timeout = 10
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            # 检查连接是否存在且可用
            if fun.s_ws is not None:
                # 检查连接状态（如果有is_connected属性）
                if hasattr(fun.s_ws, 'is_connected'):
                    if fun.s_ws.is_connected:
                        break
                else:
                    # 没有is_connected属性，假设连接可用
                    break
            time.sleep(0.1)

        if fun.s_ws is None:
            raise RuntimeError("Failed to connect to simulation server: s_ws is None")

        # 检查连接状态
        if hasattr(fun.s_ws, 'is_connected') and not fun.s_ws.is_connected:
            raise RuntimeError("Failed to connect to simulation server: connection not established")

        # n
        fun.sim_control('edit')
        time.sleep(1)
        fun.set_simulation_input(0.1)
        fun.sim_control('play', 'frameStepped')

        # I
        timeout = 5
        start_time = time.time()
        while not fun.sim_start and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        if not fun.sim_start:
            print("[VEBBattleEnv] 警告: 仿真启动超时，可能需要检查服务器状态")

    def _get_observation(self) -> Dict:
        """获取当前仿真状态观测"""
        if not self.has_sim:
            return {}

        raw_obs = copy.deepcopy(self.env.funTool.get_sim_data())
        return self._build_global_observation(raw_obs)

    def _sim_step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """e"""
        fun = self.env.funTool
        info = {}

        # \vgL
        cmds = self._decode_action(action)
        self._send_cmds(cmds)

        # gLK\
        raw_obs = copy.deepcopy(fun.get_sim_data())
        blue_side = self._get_side(raw_obs, 'blue')
        if self.opponent_agent is not None:
            blue_cmds = self.opponent_agent.get_cmds(blue_side)
            self._send_cmds(blue_cmds)

        # 
        fun.sim_data.frame = config.frame_num if hasattr(config, 'frame_num') else 10
        fun.sim_control('step')

        # I
        time.sleep(0.05)

        # ְK
        new_obs = self._get_observation()

        # V
        if self.prev_obs is not None:
            reward = self.reward_calculator.step(self.prev_obs, new_obs, {})
        else:
            reward = 0.0

        # ֶ
        next_state = self._extract_state(new_obs)

        # /&_
        done_flags = self.env.get_done()
        done = done_flags[0]

        if done:
            victory = bool(done_flags[1])
            self.reward_calculator.finalize(victory)
            info['victory'] = victory
            info['summary'] = self.reward_calculator.get_summary()

        self.prev_obs = new_obs
        return next_state, reward, done, info

    def _mock_step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Mock 环境 step"""
        action_effect = np.zeros(self.state_dim, dtype=np.float32)
        action_band = max(self.actions_per_unit, 1)
        action_idx = action % action_band
        action_effect[action_idx * 23:(action_idx + 1) * 23] = np.random.randn(23) * 0.1

        next_state = self.current_state + action_effect
        next_state = np.clip(next_state, -10, 10).astype(np.float32)

        reward = -np.sum(np.abs(next_state[:50])) * 0.01
        done = np.random.random() < 0.01 or self.step_count >= self.max_episode_steps

        info = {}
        if done:
            info['victory'] = np.random.random() < 0.5

        return next_state, reward, done, info

    def _decode_action(self, action: int) -> List[Dict]:
        """
        将离散动作映射成机动或火控指令。

        重要：为所有平台生成指令！
        - 被选中的平台执行指定动作
        - 未被选中的平台执行默认机动（朝敌方方向或保持前进）
        """
        cmds: List[Dict] = []

        if not self.has_sim:
            return cmds

        raw_obs = copy.deepcopy(self.env.funTool.get_sim_data())
        red_side = self._get_side(raw_obs, 'red')
        platforms = red_side.get('platform_list', [])[:self.max_controlled_units]

        if not platforms:
            return cmds

        # 解析主动作：选中的平台和动作
        unit_idx = action // self.actions_per_unit
        action_idx = action % self.actions_per_unit

        if unit_idx >= len(platforms):
            unit_idx = len(platforms) - 1

        # 获取敌方位置用于计算默认机动方向
        enemy_center = self._get_enemy_center(raw_obs)

        # 为所有平台生成指令
        for i, platform in enumerate(platforms):
            if i == unit_idx:
                # 被选中的平台：执行指定动作
                if action_idx < self.move_actions_per_unit:
                    cmd = self._build_move_cmd(platform, action_idx)
                else:
                    fire_slot = action_idx - self.move_actions_per_unit
                    cmd = self._build_fire_cmd(platform, fire_slot, raw_obs)
                    # 开火后也要机动，不能停着
                    if cmd:
                        cmds.append(cmd)
                    # 开火的同时继续朝敌方机动
                    cmd = self._build_move_toward_enemy(platform, enemy_center)
            else:
                # 未被选中的平台：执行默认机动（朝敌方方向）
                cmd = self._build_move_toward_enemy(platform, enemy_center)

            if cmd:
                cmds.append(cmd)

        return cmds

    def _get_enemy_center(self, raw_obs: Dict) -> Optional[Tuple[float, float]]:
        """获取敌方单位的中心位置"""
        red_side = self._get_side(raw_obs, 'red')
        track_list = red_side.get('track_list', [])

        enemy_positions = []
        for track in track_list:
            if track.get('platform_entity_side') != 'red' and track.get('platform_entity_type') != '导弹':
                lon = track.get('longitude', track.get('X'))
                lat = track.get('latitude', track.get('Y'))
                if lon is not None and lat is not None:
                    enemy_positions.append((float(lon), float(lat)))

        if not enemy_positions:
            # 没有发现敌人，返回默认目标方向（地图中心偏东）
            return (self.reward_calculator.config.center_longitude + 0.5,
                    self.reward_calculator.config.center_latitude)

        # 返回敌方中心位置
        avg_lon = sum(p[0] for p in enemy_positions) / len(enemy_positions)
        avg_lat = sum(p[1] for p in enemy_positions) / len(enemy_positions)
        return (avg_lon, avg_lat)

    def _build_move_toward_enemy(self, platform: Dict, enemy_center: Optional[Tuple[float, float]]) -> Optional[Dict]:
        """构建朝敌方方向机动的指令"""
        from utilities.yxScriptTreeFunc import YxScriptTreeFunc as decCmd

        pname = platform.get('name', platform.get('Name'))
        if not pname:
            return None

        current_lon = platform.get('longitude', platform.get('X'))
        current_lat = platform.get('latitude', platform.get('Y'))
        current_alt = platform.get('altitude', platform.get('Alt'))

        if current_lon is None or current_lat is None:
            return None

        try:
            current_lon = float(current_lon)
            current_lat = float(current_lat)
            current_alt = float(current_alt) if current_alt else 5000.0
        except (TypeError, ValueError):
            return None

        # 计算朝敌方方向的目标点
        if enemy_center:
            target_lon, target_lat = enemy_center
            # 计算方向向量并移动一定距离
            dx = target_lon - current_lon
            dy = target_lat - current_lat
            dist = (dx**2 + dy**2) ** 0.5
            if dist > 0.001:  # 避免除以零
                # 移动10km朝敌方方向
                move_dist = 10000 / 111000  # 约10km转换为度
                dx = dx / dist * move_dist
                dy = dy / dist * move_dist
            else:
                # 已经很近了，小幅度移动
                dx = 0.01
                dy = 0.01
        else:
            # 没有敌人信息，保持当前航向前进
            heading = platform.get('heading', 0)
            import math
            move_dist = 10000 / 111000
            dx = math.sin(heading) * move_dist
            dy = math.cos(heading) * move_dist

        target_lon = current_lon + dx
        target_lat = current_lat + dy

        # 边界检查
        target_lat = max(-90.0, min(90.0, target_lat))
        target_lon = max(-180.0, min(180.0, target_lon))
        current_alt = max(100.0, min(15000.0, current_alt))

        # 官方API: fly_to_point(平台名, [纬度, 经度, 高度], 速度)
        target_point = [target_lat, target_lon, current_alt]
        speed = 400  # 400 m/s

        return decCmd.fly_to_point(pname, target_point, speed)

    def _build_move_cmd(self, platform: Dict, direction_idx: int) -> Optional[Dict]:
        """构建机动指令 - 使用官方API格式"""
        from utilities.yxScriptTreeFunc import YxScriptTreeFunc as decCmd

        pname = platform.get('name', platform.get('Name'))
        if not pname:
            print("[VEBBattleEnv] _build_move_cmd: 平台名称为空")
            return None

        # 获取当前位置，确保数值有效
        current_lon = platform.get('longitude', platform.get('X'))
        current_lat = platform.get('latitude', platform.get('Y'))
        current_alt = platform.get('altitude', platform.get('Alt'))

        # 数据验证：确保坐标有效
        if current_lon is None or current_lat is None:
            print(f"[VEBBattleEnv] _build_move_cmd: 平台 {pname} 坐标无效 lon={current_lon}, lat={current_lat}")
            return None

        # 使用默认值填充缺失的高度
        if current_alt is None:
            current_alt = 5000.0

        # 确保数值类型正确
        try:
            current_lon = float(current_lon)
            current_lat = float(current_lat)
            current_alt = float(current_alt)
        except (TypeError, ValueError) as e:
            print(f"[VEBBattleEnv] _build_move_cmd: 坐标转换失败 {e}")
            return None

        directions = [
            (0, 1),    # 北
            (1, 1),    # 东北
            (1, 0),    # 东
            (1, -1),   # 东南
            (0, -1),   # 南
            (-1, -1),  # 西南
            (-1, 0),   # 西
            (-1, 1),   # 西北
        ]
        dx, dy = directions[direction_idx % len(directions)]
        move_distance = 5000  # 5km

        target_lon = current_lon + dx * move_distance / 111000
        target_lat = current_lat + dy * move_distance / 111000

        # 边界检查：确保目标点在合理范围内
        target_lat = max(-90.0, min(90.0, target_lat))
        target_lon = max(-180.0, min(180.0, target_lon))
        current_alt = max(100.0, min(15000.0, current_alt))

        # 使用官方API: fly_to_point(平台名, [纬度, 经度, 高度], 速度)
        # 注意：官方API的target_point顺序是 [纬度, 经度, 高度]
        target_point = [target_lat, target_lon, current_alt]
        speed = 400  # 默认速度 400 m/s

        return decCmd.fly_to_point(pname, target_point, speed)

    def _build_fire_cmd(self, platform: Dict, fire_slot: int, raw_obs: Dict) -> Optional[Dict]:
        """构建开火指令 - 使用官方API格式"""
        from utilities.yxScriptTreeFunc import YxScriptTreeFunc as decCmd

        pname = platform.get('name', platform.get('Name'))
        if not pname:
            print("[VEBBattleEnv] _build_fire_cmd: 平台名称为空")
            return None

        # 验证平台名是字符串
        if not isinstance(pname, str):
            print(f"[VEBBattleEnv] _build_fire_cmd: 平台名称类型错误 type={type(pname)}")
            return None

        # 获取敌方单位（从track_list中获取，这是己方能探测到的敌方单位）
        red_side = self._get_side(raw_obs, 'red')
        track_list = red_side.get('track_list', [])

        # 筛选敌方飞机（非导弹）
        enemy_aircraft = [t for t in track_list
                         if t.get('platform_entity_side') != 'red'
                         and t.get('platform_entity_type') != '导弹']

        if not enemy_aircraft:
            # 没有可攻击目标，静默返回None
            return None

        target = enemy_aircraft[fire_slot % len(enemy_aircraft)]
        target_name = target.get('target_name', target.get('name', target.get('Name')))

        if not target_name:
            print(f"[VEBBattleEnv] _build_fire_cmd: 目标名称为空, target={target}")
            return None

        # 验证目标名是字符串
        if not isinstance(target_name, str):
            print(f"[VEBBattleEnv] _build_fire_cmd: 目标名称类型错误 type={type(target_name)}")
            return None

        # 使用官方API: fire_track(平台名, 目标名)
        return decCmd.fire_track(pname, target_name)

    def _send_cmds(self, cmds: List[Dict]):
        """0"""
        if not self.has_sim:
            return

        for cmd in cmds:
            try:
                if cmd is None:
                    continue

                json_data = cmd.get('json_data')
                if json_data is None:
                    print(f"[VEBBattleEnv] 警告: cmd中没有json_data字段, cmd={cmd}")
                    continue

                # 官方API返回的json_data可能是字典或字符串
                if isinstance(json_data, dict):
                    json_str = json.dumps(json_data, ensure_ascii=False)
                else:
                    json_str = str(json_data)

                # 调试日志 - 输出发送的指令
                if hasattr(config, 'is_print_debug') and config.is_print_debug:
                    print(f"[VEBBattleEnv] 发送指令: {json_str[:200]}...")

                self.env.funTool.send_str_ws(json_str)
            except Exception as e:
                print(f"[VEBBattleEnv] 指令发送失败: {e}, cmd={cmd}")

    def _build_global_observation(self, raw_obs: Dict) -> Dict:
        """构建全局观测数据"""
        red_side = self._get_side(raw_obs, 'red')
        blue_side = self._get_side(raw_obs, 'blue')

        enemy_as_tracks = []
        for platform in blue_side.get('platform_list', []):
            enemy_as_tracks.append({
                'ID': platform.get('id', platform.get('ID')),
                'Type': 1 if platform.get('type') == '有人机' else 2,
                'X': platform.get('longitude', 0),
                'Y': platform.get('latitude', 0),
                'Alt': platform.get('altitude', 0),
                'Speed': platform.get('speed', 0),
                'Heading': platform.get('heading', 0),
                'longitude': platform.get('longitude', 0),
                'latitude': platform.get('latitude', 0),
                'altitude': platform.get('altitude', 0),
                'target_name': platform.get('name', ''),
                'platform_entity_type': platform.get('type', '无人机'),
                'platform_entity_side': 'blue',
                'is_fired_num': 0,
            })

        broken_list = []
        for side in raw_obs.get('side_list', []):
            for broken in side.get('broken_list', []):
                broken_list.append(broken)

        return {
            'platform_list': red_side.get('platform_list', []),
            'track_list': enemy_as_tracks,
            'broken_list': broken_list,
            'header': raw_obs.get('header', {})
        }

    def _get_side(self, raw_obs: Dict, side_name: str) -> Dict:
        """pn"""
        for item in raw_obs.get('side_list', []):
            if item.get('side') == side_name:
                return item
        return raw_obs.get('side_list', [{}])[0] if raw_obs.get('side_list') else {}

    def _extract_state(self, obs: Dict) -> np.ndarray:
        """ֶy"""
        state = np.zeros(self.state_dim, dtype=np.float32)

        # sy (5*UM * 10y = 50)
        for i, platform in enumerate(obs.get('platform_list', [])[:5]):
            offset = i * 10
            state[offset:offset+10] = self._extract_platform_features(platform)

        # Ly (5* * 10y = 50)
        for i, track in enumerate(obs.get('track_list', [])[:5]):
            offset = 50 + i * 10
            state[offset:offset+10] = self._extract_track_features(track)

        # h@y (130)
        state[100:230] = self._extract_global_features(obs)

        return state

    def _extract_platform_features(self, platform: Dict) -> np.ndarray:
        """sy"""
        features = np.zeros(10, dtype=np.float32)
        features[0] = platform.get('longitude', platform.get('X', 0)) / 180.0
        features[1] = platform.get('latitude', platform.get('Y', 0)) / 90.0
        features[2] = platform.get('altitude', platform.get('Alt', 0)) / 15000.0
        features[3] = platform.get('speed', platform.get('Speed', 0)) / 500.0
        features[4] = platform.get('heading', platform.get('Heading', 0)) / 360.0
        features[5] = 1.0 if platform.get('type') == '有人机' else 0.0
        features[6] = 1.0  # alive
        features[7] = platform.get('hp', 100) / 100.0
        features[8] = platform.get('fuel', 100) / 100.0
        features[9] = sum(w.get('quantity', 0) for w in platform.get('weapons', [])) / 10.0
        return features

    def _extract_track_features(self, track: Dict) -> np.ndarray:
        """y"""
        features = np.zeros(10, dtype=np.float32)
        features[0] = track.get('X', track.get('longitude', 0)) / 180.0
        features[1] = track.get('Y', track.get('latitude', 0)) / 90.0
        features[2] = track.get('Alt', track.get('altitude', 0)) / 15000.0
        features[3] = track.get('Speed', track.get('speed', 0)) / 500.0
        features[4] = track.get('Heading', track.get('heading', 0)) / 360.0
        features[5] = 1.0 if track.get('Type', 2) == 1 else 0.0
        features[6] = track.get('ThreatLevel', 0.5)
        features[7] = 0.0
        features[8] = 0.0
        features[9] = 0.0
        return features

    def _extract_global_features(self, obs: Dict) -> np.ndarray:
        """h@y"""
        features = np.zeros(130, dtype=np.float32)

        n_friendly = len(obs.get('platform_list', []))
        n_enemy = len(obs.get('track_list', []))

        features[0] = n_friendly / 5.0
        features[1] = n_enemy / 5.0
        features[2] = n_friendly / max(n_enemy, 1)
        features[3] = sum(1 for p in obs.get('platform_list', []) if p.get('type') == '有人机')
        features[4] = sum(1 for t in obs.get('track_list', []) if t.get('Type', 2) == 1)
        features[5] = obs.get('header', {}).get('sim_time', 0) / 1000.0
        features[6] = self.step_count / self.max_episode_steps

        return features

    def close(self):
        """关闭环境，释放所有资源"""
        if self.has_sim and self.env is not None:
            # 停止推演
            try:
                self.env.funTool.sim_control('edit')
            except:
                pass

            # 关闭websocket连接
            try:
                if hasattr(self.env, 'socket_manager'):
                    self.env.socket_manager.close_all()
            except Exception as e:
                print(f"[VEBBattleEnv] 关闭socket_manager时出错: {e}")

            # 调用Env的close方法（如果存在）
            try:
                if hasattr(self.env, 'close'):
                    self.env.close()
            except Exception as e:
                print(f"[VEBBattleEnv] 关闭Env时出错: {e}")

        self.episode_started = False
        self.env = None


# K
if __name__ == "__main__":
    print("Testing VEBBattleEnvironment...")

    # !!	
    env = VEBBattleEnvironment(
        state_dim=230,
        action_dim=40,
        max_episode_steps=100
    )

    # K reset
    state = env.reset()
    print(f"Initial state shape: {state.shape}")

    # K step
    total_reward = 0
    for i in range(20):
        action = np.random.randint(0, 40)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        print(f"Step {i+1}: reward={reward:.4f}, done={done}")
        if done:
            break

    print(f"\nTotal reward: {total_reward:.4f}")
    env.close()
