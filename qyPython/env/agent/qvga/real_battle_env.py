"""
真实仿真环境对战接口

用于QVGA在线训练，与真实仿真环境交互
"""
import os
import sys
import time
import copy
import json
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

import config
from env.env import Env, auto_engage_main
from .individual import Individual, PolicyNetwork, decode_weights
from .reward import RewardCalculator, RewardConfig


class RealBattleEnvironment:
    """
    真实仿真环境对战接口

    将QVGA个体与仿真环境连接，进行在线训练
    """

    def __init__(self, reward_config: Optional[RewardConfig] = None, device: str = 'cpu'):
        """
        Args:
            reward_config: 奖励函数配置
            device: 计算设备
        """
        self.reward_config = reward_config or RewardConfig()
        self.device = torch.device(device)

        # 策略网络
        self.policy_network = PolicyNetwork().to(self.device)

        # 奖励计算器
        self.reward_calculator = RewardCalculator(self.reward_config)

        # 状态维度
        self.state_dim = 230
        self.action_dim = 40

        # 对战记录
        self.battle_history: List[Dict] = []

    def battle(self, individual: Individual, opponent_agent=None) -> Dict:
        """
        执行一场完整对战

        Args:
            individual: QVGA个体
            opponent_agent: 对手Agent（默认使用DemoAutoAgent）

        Returns:
            对战结果字典
        """
        # 载入个体权重到策略网络
        decode_weights(individual.weights, self.policy_network)

        # 创建QVGA Agent包装器
        qvga_agent = QVGABattleAgent(self.policy_network, self.device)

        # 获取对手
        if opponent_agent is None:
            from env.agent.demo.demo_auto_agent import DemoAutoAgent
            opponent_agent = DemoAutoAgent()

        # 初始化结果
        result = {
            'win': 0,
            'loss': 0,
            'reward': 0.0,
            'states': [],
            'experiences': [],
            'battle_stats': {}
        }

        try:
            # 执行对战
            battle_result = self._run_battle(qvga_agent, opponent_agent)

            # 解析结果
            result['win'] = 1 if battle_result['victory'] else 0
            result['loss'] = 0 if battle_result['victory'] else 1
            result['reward'] = battle_result['total_reward']
            result['states'] = battle_result['states']
            result['experiences'] = battle_result['experiences']
            result['battle_stats'] = battle_result['stats']

        except Exception as e:
            print(f"对战执行出错: {e}")
            result['loss'] = 1
            result['reward'] = -50.0

        return result

    def _run_battle(self, qvga_agent, opponent_agent) -> Dict:
        """
        运行单场对战

        这是一个简化版本，实际需要与仿真环境集成
        """
        states = []
        experiences = []
        prev_obs = None

        # 重置奖励计算器
        self.reward_calculator.reset()

        # 模拟对战步骤（实际应该与Env集成）
        max_steps = 500
        step_count = 0

        for step in range(max_steps):
            # 获取当前观测（这里需要从仿真环境获取）
            curr_obs = self._get_observation()

            if prev_obs is None:
                # 第一步，初始化状态
                self.reward_calculator.reset(curr_obs)
                prev_obs = curr_obs
                continue

            # 提取状态特征
            state = self._extract_state(curr_obs)
            states.append(state)

            # 获取动作
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action = self.policy_network(state_t)
                action = action.cpu().numpy().flatten()

            # 执行动作（发送到仿真环境）
            actions = self._decode_action(action, curr_obs)

            # 计算即时奖励
            step_reward = self.reward_calculator.step(prev_obs, curr_obs, actions)

            # 检查是否结束
            done = self._check_done(curr_obs)

            # 下一状态
            next_obs = self._get_observation()
            next_state = self._extract_state(next_obs)

            # 记录经验
            experiences.append({
                'state': state,
                'policy_weights': individual.weights if hasattr(self, 'current_individual') else np.zeros(1),
                'reward': step_reward,
                'next_state': next_state,
                'done': done
            })

            prev_obs = curr_obs
            step_count += 1

            if done:
                break

        # 计算终局奖励
        victory = self._check_victory(curr_obs)
        final_reward = self.reward_calculator.finalize(victory)

        # 获取统计
        summary = self.reward_calculator.get_summary()

        return {
            'victory': victory,
            'total_reward': summary['total_reward'],
            'states': states,
            'experiences': experiences,
            'stats': summary['stats'],
            'steps': step_count
        }

    def _get_observation(self) -> Dict:
        """
        从仿真环境获取观测

        需要与Env类集成
        """
        # 这里返回模拟数据，实际应该从env.funTool.get_sim_data()获取
        return {
            'platform_list': [],
            'track_list': [],
            'header': {'sim_time': 0}
        }

    def _extract_state(self, obs: Dict) -> np.ndarray:
        """
        从观测中提取状态特征

        状态向量结构（共230维）:
        - 己方单位特征: 5单位 × 10特征 = 50维
        - 敌方单位特征: 5单位 × 10特征 = 50维
        - 己方导弹状态: 5单位 × 6特征 = 30维
        - 全局态势特征: 100维
        """
        state = np.zeros(self.state_dim, dtype=np.float32)

        # 己方单位特征 (0-49)
        if 'platform_list' in obs:
            for i, platform in enumerate(obs['platform_list'][:5]):
                offset = i * 10
                state[offset:offset+10] = self._extract_platform_features(platform)

        # 敌方单位特征 (50-99)
        if 'track_list' in obs:
            for i, track in enumerate(obs['track_list'][:5]):
                offset = 50 + i * 10
                state[offset:offset+10] = self._extract_track_features(track)

        # 导弹状态 (100-129)
        if 'platform_list' in obs:
            for i, platform in enumerate(obs['platform_list'][:5]):
                offset = 100 + i * 6
                state[offset:offset+6] = self._extract_missile_features(platform)

        # 全局态势特征 (130-229)
        state[130:230] = self._extract_global_features(obs)

        return state

    def _extract_platform_features(self, platform: Dict) -> np.ndarray:
        """提取单个己方平台特征（10维）"""
        features = np.zeros(10, dtype=np.float32)

        # 位置 (归一化)
        features[0] = platform.get('X', 0) / 100000.0  # x坐标
        features[1] = platform.get('Y', 0) / 100000.0  # y坐标
        features[2] = platform.get('Alt', platform.get('Z', 0)) / 10000.0  # 高度

        # 速度
        features[3] = platform.get('Speed', 0) / 500.0  # 速度

        # 航向
        features[4] = platform.get('Heading', 0) / 360.0  # 航向

        # 类型 (one-hot: 有人机=1, 无人机=0)
        features[5] = 1.0 if platform.get('Type', 0) == 1 else 0.0

        # 存活状态
        features[6] = 1.0  # 存活

        # HP/燃油（如果有）
        features[7] = platform.get('HP', 100) / 100.0
        features[8] = platform.get('Fuel', 100) / 100.0

        # 剩余导弹数
        features[9] = platform.get('MissileCount', 4) / 4.0

        return features

    def _extract_track_features(self, track: Dict) -> np.ndarray:
        """提取单个敌方目标特征（10维）"""
        features = np.zeros(10, dtype=np.float32)

        # 位置
        features[0] = track.get('X', 0) / 100000.0
        features[1] = track.get('Y', 0) / 100000.0
        features[2] = track.get('Alt', track.get('Z', 0)) / 10000.0

        # 速度
        features[3] = track.get('Speed', 0) / 500.0

        # 航向
        features[4] = track.get('Heading', 0) / 360.0

        # 类型
        features[5] = 1.0 if track.get('Type', 0) == 1 else 0.0

        # 威胁等级
        features[6] = track.get('ThreatLevel', 0.5)

        # 被锁定状态
        features[7] = 1.0 if track.get('is_fired_num', 0) > 0 else 0.0

        # 距离最近己方单位的距离
        features[8] = track.get('MinDistance', 50000) / 100000.0

        # 预留
        features[9] = 0.0

        return features

    def _extract_missile_features(self, platform: Dict) -> np.ndarray:
        """提取导弹状态特征（6维）"""
        features = np.zeros(6, dtype=np.float32)

        # 剩余导弹数量
        features[0] = platform.get('MissileCount', 4) / 4.0

        # 导弹类型分布（如果有多种）
        features[1] = platform.get('AAMCount', 2) / 4.0  # 空空导弹
        features[2] = platform.get('AGMCount', 2) / 4.0  # 空地导弹

        # 是否有导弹在飞行中
        features[3] = 1.0 if platform.get('MissileInFlight', 0) > 0 else 0.0

        # 冷却状态
        features[4] = platform.get('WeaponReady', 1.0)

        # 预留
        features[5] = 0.0

        return features

    def _extract_global_features(self, obs: Dict) -> np.ndarray:
        """提取全局态势特征（100维）"""
        features = np.zeros(100, dtype=np.float32)

        # 己方存活数量
        own_count = len(obs.get('platform_list', []))
        features[0] = own_count / 5.0

        # 敌方存活数量
        enemy_count = len(obs.get('track_list', []))
        features[1] = enemy_count / 5.0

        # 数量比
        features[2] = own_count / max(enemy_count, 1)

        # 己方有人机存活
        own_manned = sum(1 for p in obs.get('platform_list', []) if p.get('Type', 0) == 1)
        features[3] = own_manned

        # 敌方有人机存活
        enemy_manned = sum(1 for t in obs.get('track_list', []) if t.get('Type', 0) == 1)
        features[4] = enemy_manned

        # 时间（如果有）
        features[5] = obs.get('header', {}).get('sim_time', 0) / 1000.0

        # 其余特征可以用于更复杂的态势编码
        # 例如：区域控制、威胁分布、战术态势等

        return features

    def _decode_action(self, action: np.ndarray, obs: Dict) -> Dict:
        """
        将网络输出解码为具体指令

        action: 40维向量
        - 每个单位8维: [move_x, move_y, move_z, speed, fire_target, fire_prob, evade, formation]
        """
        actions = {}
        platforms = obs.get('platform_list', [])

        for i, platform in enumerate(platforms[:5]):
            offset = i * 8
            unit_action = action[offset:offset+8]

            pid = platform.get('ID', platform.get('id', i))
            actions[pid] = {
                'move_direction': (unit_action[0], unit_action[1]),  # 移动方向
                'altitude_change': unit_action[2],                    # 高度变化
                'speed_change': unit_action[3],                       # 速度变化
                'fire_target': int(unit_action[4] * 5),              # 攻击目标索引
                'fire_probability': unit_action[5],                   # 发射概率
                'evade': unit_action[6] > 0.5,                       # 是否规避
                'formation': int(unit_action[7] * 3)                 # 编队模式
            }

        return actions

    def _check_done(self, obs: Dict) -> bool:
        """检查对战是否结束"""
        # 己方有人机全灭
        own_manned = [p for p in obs.get('platform_list', []) if p.get('Type', 0) == 1]
        if len(own_manned) == 0:
            return True

        # 敌方有人机全灭
        enemy_manned = [t for t in obs.get('track_list', []) if t.get('Type', 0) == 1]
        if len(enemy_manned) == 0:
            return True

        # 超时
        sim_time = obs.get('header', {}).get('sim_time', 0)
        if sim_time > 600:  # 10分钟超时
            return True

        return False

    def _check_victory(self, obs: Dict) -> bool:
        """检查是否胜利"""
        # 敌方有人机全灭且己方有人机存活
        own_manned = [p for p in obs.get('platform_list', []) if p.get('Type', 0) == 1]
        enemy_manned = [t for t in obs.get('track_list', []) if t.get('Type', 0) == 1]

        return len(enemy_manned) == 0 and len(own_manned) > 0


class QVGABattleAgent:
    """
    QVGA对战Agent包装器

    将PolicyNetwork包装成可与仿真环境交互的Agent
    """

    def __init__(self, policy_network: PolicyNetwork, device: torch.device):
        self.policy_network = policy_network
        self.device = device
        self.state_extractor = RealBattleEnvironment()

    def get_cmds(self, side_data: Dict) -> List[Dict]:
        """
        获取指令（与仿真环境接口兼容）

        Args:
            side_data: 己方态势数据

        Returns:
            指令列表
        """
        # 提取状态
        state = self.state_extractor._extract_state(side_data)

        # 获取动作
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.policy_network(state_t)
            action = action.cpu().numpy().flatten()

        # 解码为指令
        cmds = self._action_to_cmds(action, side_data)

        return cmds

    def _action_to_cmds(self, action: np.ndarray, side_data: Dict) -> List[Dict]:
        """将动作向量转换为仿真指令"""
        cmds = []
        platforms = side_data.get('platform_list', [])
        tracks = side_data.get('track_list', [])

        for i, platform in enumerate(platforms[:5]):
            offset = i * 8
            unit_action = action[offset:offset+8]

            pid = platform.get('ID', platform.get('id', i))
            pname = platform.get('Name', platform.get('name', f'unit_{i}'))

            # 移动指令
            if abs(unit_action[0]) > 0.1 or abs(unit_action[1]) > 0.1:
                # 计算目标位置
                current_x = platform.get('X', 0)
                current_y = platform.get('Y', 0)
                target_x = current_x + unit_action[0] * 10000  # 移动距离
                target_y = current_y + unit_action[1] * 10000

                move_cmd = {
                    'json_data': json.dumps({
                        'fun': 'SetPath',
                        'Name': pname,
                        'Points': [{'X': target_x, 'Y': target_y, 'Alt': platform.get('Alt', 5000)}]
                    })
                }
                cmds.append(move_cmd)

            # 攻击指令
            fire_prob = unit_action[5]
            if fire_prob > 0.5 and len(tracks) > 0:
                target_idx = min(int(unit_action[4] * len(tracks)), len(tracks) - 1)
                target = tracks[target_idx]
                target_name = target.get('Name', target.get('name', ''))

                fire_cmd = {
                    'json_data': json.dumps({
                        'fun': 'FireAtTrack',
                        'Name': pname,
                        'TargetName': target_name
                    })
                }
                cmds.append(fire_cmd)

            # 规避指令
            if unit_action[6] > 0.5:
                evade_cmd = {
                    'json_data': json.dumps({
                        'fun': 'Evade',
                        'Name': pname
                    })
                }
                cmds.append(evade_cmd)

        return cmds


# 测试代码
if __name__ == "__main__":
    # 创建真实环境
    env = RealBattleEnvironment()

    # 创建测试个体
    from .individual import Individual
    individual = Individual()

    # 执行对战测试
    print("Testing RealBattleEnvironment...")
    result = env.battle(individual)
    print(f"Result: win={result['win']}, loss={result['loss']}, reward={result['reward']:.2f}")
    print(f"States collected: {len(result['states'])}")
    print(f"Experiences collected: {len(result['experiences'])}")
