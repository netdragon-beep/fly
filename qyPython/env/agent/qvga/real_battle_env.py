"""
真实仿真环境对接与在线训练接口。

该模块封装 Env，与 YxScriptTreeFunc 交互，方便在真实空战仿真中
运行 QVGA 个体、采集经验并返回奖励。
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

# 添加项目路径，确保运行 CLI / VSCode 等环境时能定位到 repo 根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

import config
from env.env import Env
from env.agent.demo.demo_auto_agent import DemoAutoAgent
from .individual import Individual, PolicyNetwork, decode_weights
from .reward import RewardCalculator, RewardConfig


class RealBattleEnvironment:
    """
    负责和真实仿真服务器交互的控制器。

    - 建立 WebSocket 连接，调度仿真推演
    - 调用 QVGA 策略生成红方指令
    - 调用对手机制（默认 DemoAutoAgent）生成蓝方指令
    - 每一步根据观测计算奖励，输出经验
    """

    def __init__(self,
                 reward_config: Optional[RewardConfig] = None,
                 device: str = 'cpu',
                 opponent_agent: Optional[DemoAutoAgent] = None):
        self.device = torch.device(device)
        self.policy_network = PolicyNetwork().to(self.device)
        self.reward_calculator = RewardCalculator(reward_config or RewardConfig())

        # 建立到仿真服务器的连接；这里不把 agent 交给 Env，由本类手动调度
        self.env = Env(red_agent=None, blue_agent=None)
        self.opponent_agent = opponent_agent or DemoAutoAgent('blue', 'blue_demo')

    # ------------------------------------------------------------------ #
    # 公开接口
    # ------------------------------------------------------------------ #
    def battle(self,
               individual: Individual,
               opponent_agent: Optional[DemoAutoAgent] = None) -> Dict:
        """
        执行一场完整的仿真对战，并返回奖励/经验等信息。
        """
        decode_weights(individual.weights, self.policy_network)
        qvga_agent = QVGABattleAgent(self.policy_network, self.device)
        blue_agent = opponent_agent or self.opponent_agent

        try:
            battle_summary = self._run_episode(qvga_agent, blue_agent, individual.weights)
        except Exception as exc:  # 线上训练不希望因为异常中断
            print(f"[RealBattleEnv] 对战发生异常: {exc}")
            battle_summary = {
                'victory': False,
                'total_reward': -50.0,
                'states': [],
                'experiences': [],
                'stats': {},
            }

        return {
            'win': 1 if battle_summary['victory'] else 0,
            'loss': 0 if battle_summary['victory'] else 1,
            'reward': battle_summary['total_reward'],
            'states': battle_summary['states'],
            'experiences': battle_summary['experiences'],
            'battle_stats': battle_summary['stats'],
        }

    # ------------------------------------------------------------------ #
    # 核心控制逻辑
    # ------------------------------------------------------------------ #
    def _run_episode(self,
                     red_agent: "QVGABattleAgent",
                     blue_agent: DemoAutoAgent,
                     policy_weights: np.ndarray) -> Dict:
        """
        仿照 auto_engage_main 的流程逐步推进仿真，并在过程中记录奖励。
        """
        fun = self.env.funTool
        done_flags = [False, False, False]
        start_once = True

        prev_obs = None
        states: List[np.ndarray] = []
        experiences: List[Dict] = []

        self.reward_calculator.reset()

        while not done_flags[0]:
            if fun.s_ws is None:
                time.sleep(0.05)
                continue

            if not fun.sim_start:
                if start_once:
                    fun.sim_control('edit')
                    time.sleep(2)
                    fun.set_simulation_input(0.1)
                    fun.sim_control('play', 'frameStepped')
                    start_once = False
                else:
                    time.sleep(0.02)
                continue

            raw_observation = copy.deepcopy(fun.get_sim_data())

            # 进入新一帧时解析奖励、执行动作
            if fun.sim_data.frame == 0:
                global_obs = self._build_global_observation(raw_observation)

                if prev_obs is None:
                    self.reward_calculator.reset(global_obs)
                else:
                    step_reward = self.reward_calculator.step(prev_obs, global_obs, {})
                    state = self._extract_state(prev_obs)
                    next_state = self._extract_state(global_obs)
                    experiences.append({
                        'state': state,
                        'policy_weights': policy_weights,
                        'reward': step_reward,
                        'next_state': next_state,
                        'done': False
                    })
                    states.append(state)

                red_side = self._get_side(raw_observation, 'red')
                blue_side = self._get_side(raw_observation, 'blue')
                red_cmds = red_agent.get_cmds(red_side)
                blue_cmds = blue_agent.get_cmds(blue_side)

                self._send_cmds(red_cmds + blue_cmds)

                # 推演一步
                fun.sim_data.frame = config.frame_num
                fun.sim_control('step')

                prev_obs = global_obs
                done_flags = self.env.get_done()
            else:
                time.sleep(0.02)

        # 推演结束，计算终局奖励
        victory = bool(done_flags[1])
        self.reward_calculator.finalize(victory)
        summary = self.reward_calculator.get_summary()

        # 标记最后一个 transition 的 done
        if experiences:
            experiences[-1]['done'] = True

        fun.sim_control('edit')

        return {
            'victory': victory,
            'total_reward': summary['total_reward'],
            'states': states,
            'experiences': experiences,
            'stats': summary['stats']
        }

    # ------------------------------------------------------------------ #
    # 工具函数
    # ------------------------------------------------------------------ #
    def _build_global_observation(self, raw_obs: Dict) -> Dict:
        """
        将 env.funTool 返回的 raw 数据转换为 RewardCalculator 需要的格式。
        """
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
                # 奖励计算需要的额外字段
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
        for item in raw_obs.get('side_list', []):
            if item.get('side') == side_name:
                return item
        return raw_obs.get('side_list', [{}])[0]

    def _send_cmds(self, cmds: List[Dict]):
        for cmd in cmds:
            try:
                self.env.funTool.send_str_ws(cmd['json_data'])
            except Exception as exc:
                print(f"[RealBattleEnv] 指令发送失败: {exc}")

    # ------------------------------------------------------------------ #
    # 状态编码（借用原 QVGA 解析逻辑）
    # ------------------------------------------------------------------ #
    def _extract_state(self, obs: Dict) -> np.ndarray:
        state = np.zeros(230, dtype=np.float32)
        for i, platform in enumerate(obs.get('platform_list', [])[:5]):
            offset = i * 10
            state[offset:offset+10] = self._extract_platform_features(platform)
        for i, track in enumerate(obs.get('track_list', [])[:5]):
            offset = 50 + i * 10
            state[offset:offset+10] = self._extract_track_features(track)
        state[130:230] = self._extract_global_features(obs)
        return state

    def _extract_platform_features(self, platform: Dict) -> np.ndarray:
        features = np.zeros(10, dtype=np.float32)
        features[0] = platform.get('longitude', platform.get('X', 0)) / 100
        features[1] = platform.get('latitude', platform.get('Y', 0)) / 100
        features[2] = platform.get('altitude', platform.get('Alt', 0)) / 10000
        features[3] = platform.get('speed', platform.get('Speed', 0)) / 500
        features[4] = platform.get('heading', platform.get('Heading', 0)) / 360
        features[5] = 1.0 if platform.get('type') == '有人机' else 0.0
        features[6] = 1.0
        features[7] = platform.get('hp', 100) / 100
        features[8] = platform.get('fuel', 100) / 100
        features[9] = sum(w.get('quantity', 0) for w in platform.get('weapons', [])) / 10
        return features

    def _extract_track_features(self, track: Dict) -> np.ndarray:
        features = np.zeros(10, dtype=np.float32)
        features[0] = track.get('X', 0) / 100
        features[1] = track.get('Y', 0) / 100
        features[2] = track.get('Alt', 0) / 10000
        features[3] = track.get('Speed', 0) / 500
        features[4] = track.get('Heading', 0) / 360
        features[5] = 1.0 if track.get('Type', 2) == 1 else 0.0
        features[6] = track.get('ThreatLevel', 0.5)
        features[7] = 0.0
        features[8] = 0.0
        features[9] = 0.0
        return features

    def _extract_global_features(self, obs: Dict) -> np.ndarray:
        features = np.zeros(100, dtype=np.float32)
        features[0] = len(obs.get('platform_list', [])) / 5
        features[1] = len(obs.get('track_list', [])) / 5
        features[2] = features[0] / max(features[1], 0.1)
        features[3] = sum(1 for p in obs.get('platform_list', []) if p.get('type') == '有人机')
        features[4] = sum(1 for t in obs.get('track_list', []) if t.get('Type', 2) == 1)
        features[5] = obs.get('header', {}).get('sim_time', 0) / 1000
        return features

    # ------------------------------------------------------------------ #
    # 对战代理，从网络输出到仿真指令
    # ------------------------------------------------------------------ #
    def _build_command(self, platform_name: str, target_point: Tuple[float, float, float]) -> Dict:
        return {
            'json_data': json.dumps({
                'fun': 'SetPath',
                'Name': platform_name,
                'Points': [{'X': target_point[0], 'Y': target_point[1], 'Alt': target_point[2]}]
            })
        }

    @property
    def policy_network(self) -> PolicyNetwork:
        return self._policy_network

    @policy_network.setter
    def policy_network(self, network: PolicyNetwork):
        self._policy_network = network


class QVGABattleAgent:
    """
    将策略网络输出转换成仿真指令的封装。
    """

    def __init__(self, policy_network: PolicyNetwork, device: torch.device):
        self.policy_network = policy_network
        self.device = device

    def get_cmds(self, side_data: Dict) -> List[Dict]:
        state = self._extract_state(side_data)
        with torch.no_grad():
            state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
            actions = self.policy_network(state_t).cpu().numpy().flatten()
        return self._actions_to_cmds(actions, side_data)

    def _extract_state(self, side_data: Dict) -> np.ndarray:
        obs = {
            'platform_list': side_data.get('platform_list', []),
            'track_list': side_data.get('track_list', []),
            'header': {}
        }
        features = np.zeros(230, dtype=np.float32)
        for i, platform in enumerate(obs['platform_list'][:5]):
            offset = i * 10
            features[offset:offset+10] = self._encode_platform(platform)
        for i, track in enumerate(obs['track_list'][:5]):
            offset = 50 + i * 10
            features[offset:offset+10] = self._encode_track(track)
        return features

    def _encode_platform(self, platform: Dict) -> np.ndarray:
        feats = np.zeros(10, dtype=np.float32)
        feats[0] = platform.get('longitude', 0) / 100
        feats[1] = platform.get('latitude', 0) / 100
        feats[2] = platform.get('altitude', 0) / 10000
        feats[3] = platform.get('speed', 0) / 500
        feats[4] = platform.get('heading', 0) / 360
        feats[5] = 1.0 if platform.get('type') == '有人机' else 0.0
        feats[6] = 1.0
        feats[7] = sum(w.get('quantity', 0) for w in platform.get('weapons', [])) / 10
        feats[8] = 0.0
        feats[9] = 0.0
        return feats

    def _encode_track(self, track: Dict) -> np.ndarray:
        feats = np.zeros(10, dtype=np.float32)
        feats[0] = track.get('longitude', track.get('X', 0)) / 100
        feats[1] = track.get('latitude', track.get('Y', 0)) / 100
        feats[2] = track.get('altitude', track.get('Alt', 0)) / 10000
        feats[3] = track.get('speed', track.get('Speed', 0)) / 500
        feats[4] = track.get('heading', track.get('Heading', 0)) / 360
        feats[5] = 1.0 if track.get('type') == '有人机' else 0.0
        return feats

    def _actions_to_cmds(self, action: np.ndarray, side_data: Dict) -> List[Dict]:
        cmds = []
        for i, platform in enumerate(side_data.get('platform_list', [])[:5]):
            offset = i * 8
            unit_action = action[offset:offset+8]
            pname = platform.get('name', platform.get('Name', f'unit_{i}'))
            current_lon = platform.get('longitude', platform.get('X', 0))
            current_lat = platform.get('latitude', platform.get('Y', 0))
            current_alt = platform.get('altitude', platform.get('Alt', 5000))
            target_lon = current_lon + unit_action[0] * 10000
            target_lat = current_lat + unit_action[1] * 10000
            target_alt = current_alt + unit_action[2] * 2000
            move_cmd = {
                'json_data': json.dumps({
                    'fun': 'SetPath',
                    'Name': pname,
                    'Points': [{'X': target_lon, 'Y': target_lat, 'Alt': target_alt}]
                })
            }
            cmds.append(move_cmd)

            fire_prob = unit_action[5]
            if fire_prob > 0.5 and side_data.get('track_list'):
                target_idx = min(int(abs(unit_action[4]) * len(side_data['track_list'])),
                                 len(side_data['track_list']) - 1)
                target = side_data['track_list'][target_idx]
                fire_cmd = {
                    'json_data': json.dumps({
                        'fun': 'FireAtTrack',
                        'Name': pname,
                        'TargetName': target.get('name', target.get('target_name', ''))
                    })
                }
                cmds.append(fire_cmd)
        return cmds


if __name__ == "__main__":
    env = RealBattleEnvironment(device='cpu')
    test_individual = Individual()
    print("Running test battle ...")
    result = env.battle(test_individual)
    print(f"win={result['win']} reward={result['reward']:.2f} experiences={len(result['experiences'])}")
