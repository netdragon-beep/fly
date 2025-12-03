"""
奖励函数模块

设计原则:
1. 有人机价值 > 无人机价值
2. 击毁敌方 > 保护己方（进攻性策略）
3. 导弹精准使用，避免浪费
4. 坠机惩罚，避免自杀式策略
"""
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field


@dataclass
class RewardConfig:
    """奖励函数配置"""

    # === 击毁奖励 ===
    kill_enemy_manned: float = 30.0      # 击毁敌方有人机
    kill_enemy_uav: float = 10.0         # 击毁敌方无人机

    # === 损失惩罚 ===
    lose_own_manned: float = -25.0       # 己方有人机被击落
    lose_own_uav: float = -8.0           # 己方无人机被击落

    # === 导弹相关 ===
    missile_fired: float = -1.0          # 发射导弹成本
    missile_hit: float = 5.0             # 导弹命中额外奖励
    missile_miss: float = -2.0           # 导弹未命中惩罚

    # === 坠机惩罚 ===
    crash_penalty: float = -10.0         # 非战斗坠机惩罚

    # === 胜负奖励 ===
    victory_bonus: float = 50.0          # 胜利奖励
    defeat_penalty: float = -30.0        # 失败惩罚

    # === 辅助奖励（可选）===
    protect_manned_bonus: float = 0.5    # 每步有人机存活奖励
    damage_dealt: float = 0.1            # 造成伤害奖励（每点HP）
    survival_bonus: float = 0.01         # 存活时间奖励（每步）


@dataclass
class BattleState:
    """战场状态跟踪"""

    # 存活单位ID集合
    own_manned_alive: Set[int] = field(default_factory=set)
    own_uav_alive: Set[int] = field(default_factory=set)
    enemy_manned_alive: Set[int] = field(default_factory=set)
    enemy_uav_alive: Set[int] = field(default_factory=set)

    # 发射的导弹 {missile_id: (source_id, target_id, fire_time)}
    missiles_in_flight: Dict[int, Tuple[int, int, float]] = field(default_factory=dict)

    # 统计
    total_missiles_fired: int = 0
    total_missiles_hit: int = 0
    total_missiles_miss: int = 0

    # 击杀统计
    enemy_manned_killed: int = 0
    enemy_uav_killed: int = 0
    own_manned_lost: int = 0
    own_uav_lost: int = 0
    own_crashed: int = 0  # 非战斗坠机


class RewardCalculator:
    """
    奖励计算器

    使用方法:
    1. 每场对战开始时调用 reset()
    2. 每步调用 step(current_obs, actions) 获取即时奖励
    3. 对战结束时调用 finalize() 获取终局奖励
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        self.state = BattleState()
        self.step_count = 0
        self.total_reward = 0.0
        self.reward_breakdown: Dict[str, float] = {}

    def reset(self, initial_obs: Dict = None):
        """重置状态（新对战开始）"""
        self.state = BattleState()
        self.step_count = 0
        self.total_reward = 0.0
        self.reward_breakdown = {
            'kill_manned': 0.0,
            'kill_uav': 0.0,
            'lose_manned': 0.0,
            'lose_uav': 0.0,
            'missile_cost': 0.0,
            'missile_hit': 0.0,
            'missile_miss': 0.0,
            'crash': 0.0,
            'victory': 0.0,
            'auxiliary': 0.0
        }

        # 如果提供初始观测，解析初始状态
        if initial_obs:
            self._parse_initial_state(initial_obs)

    def _parse_initial_state(self, obs: Dict):
        """解析初始战场状态"""
        # 解析己方单位
        if 'platform_list' in obs:
            for platform in obs['platform_list']:
                pid = platform.get('ID', platform.get('id', 0))
                ptype = platform.get('Type', platform.get('type', 0))

                # Type: 1=有人机, 2=无人机 (根据实际定义调整)
                if ptype == 1:  # 有人机
                    self.state.own_manned_alive.add(pid)
                else:  # 无人机
                    self.state.own_uav_alive.add(pid)

        # 解析敌方单位
        if 'track_list' in obs:
            for track in obs['track_list']:
                tid = track.get('ID', track.get('id', 0))
                ttype = track.get('Type', track.get('type', 0))

                if ttype == 1:  # 有人机
                    self.state.enemy_manned_alive.add(tid)
                else:  # 无人机
                    self.state.enemy_uav_alive.add(tid)

    def step(self, prev_obs: Dict, curr_obs: Dict, actions: Dict = None) -> float:
        """
        计算单步奖励

        Args:
            prev_obs: 上一步观测
            curr_obs: 当前观测
            actions: 本步执行的动作

        Returns:
            即时奖励
        """
        reward = 0.0
        self.step_count += 1

        # 1. 检测敌方单位被击毁
        reward += self._check_enemy_kills(prev_obs, curr_obs)

        # 2. 检测己方损失
        reward += self._check_own_losses(prev_obs, curr_obs)

        # 3. 检测导弹发射和命中
        reward += self._check_missiles(prev_obs, curr_obs, actions)

        # 4. 检测坠机
        reward += self._check_crashes(prev_obs, curr_obs)

        # 5. 辅助奖励
        reward += self._compute_auxiliary_rewards(curr_obs)

        self.total_reward += reward
        return reward

    def _check_enemy_kills(self, prev_obs: Dict, curr_obs: Dict) -> float:
        """检测敌方单位被击毁"""
        reward = 0.0

        prev_tracks = set()
        curr_tracks = set()

        # 获取上一步敌方存活单位
        if 'track_list' in prev_obs:
            for track in prev_obs['track_list']:
                tid = track.get('ID', track.get('id', 0))
                prev_tracks.add(tid)

        # 获取当前敌方存活单位
        if 'track_list' in curr_obs:
            for track in curr_obs['track_list']:
                tid = track.get('ID', track.get('id', 0))
                curr_tracks.add(tid)

        # 检测被击毁的敌方单位
        killed = prev_tracks - curr_tracks

        # 也检查broken_list
        if 'broken_list' in curr_obs:
            for broken in curr_obs['broken_list']:
                bid = broken.get('ID', broken.get('id', 0))
                if bid in self.state.enemy_manned_alive or bid in self.state.enemy_uav_alive:
                    killed.add(bid)

        for tid in killed:
            if tid in self.state.enemy_manned_alive:
                reward += self.config.kill_enemy_manned
                self.state.enemy_manned_alive.discard(tid)
                self.state.enemy_manned_killed += 1
                self.reward_breakdown['kill_manned'] += self.config.kill_enemy_manned
                print(f"  [REWARD] 击毁敌方有人机 +{self.config.kill_enemy_manned}")

            elif tid in self.state.enemy_uav_alive:
                reward += self.config.kill_enemy_uav
                self.state.enemy_uav_alive.discard(tid)
                self.state.enemy_uav_killed += 1
                self.reward_breakdown['kill_uav'] += self.config.kill_enemy_uav
                print(f"  [REWARD] 击毁敌方无人机 +{self.config.kill_enemy_uav}")

        return reward

    def _check_own_losses(self, prev_obs: Dict, curr_obs: Dict) -> float:
        """检测己方损失"""
        reward = 0.0

        prev_platforms = set()
        curr_platforms = set()

        if 'platform_list' in prev_obs:
            for p in prev_obs['platform_list']:
                prev_platforms.add(p.get('ID', p.get('id', 0)))

        if 'platform_list' in curr_obs:
            for p in curr_obs['platform_list']:
                curr_platforms.add(p.get('ID', p.get('id', 0)))

        lost = prev_platforms - curr_platforms

        for pid in lost:
            if pid in self.state.own_manned_alive:
                reward += self.config.lose_own_manned
                self.state.own_manned_alive.discard(pid)
                self.state.own_manned_lost += 1
                self.reward_breakdown['lose_manned'] += self.config.lose_own_manned
                print(f"  [REWARD] 己方有人机被击落 {self.config.lose_own_manned}")

            elif pid in self.state.own_uav_alive:
                reward += self.config.lose_own_uav
                self.state.own_uav_alive.discard(pid)
                self.state.own_uav_lost += 1
                self.reward_breakdown['lose_uav'] += self.config.lose_own_uav
                print(f"  [REWARD] 己方无人机被击落 {self.config.lose_own_uav}")

        return reward

    def _check_missiles(self, prev_obs: Dict, curr_obs: Dict, actions: Dict) -> float:
        """检测导弹发射和命中"""
        reward = 0.0

        # 检测导弹发射（从actions中）
        if actions:
            for unit_actions in actions.values():
                if isinstance(unit_actions, dict):
                    if unit_actions.get('fire_track') or unit_actions.get('fire'):
                        reward += self.config.missile_fired
                        self.state.total_missiles_fired += 1
                        self.reward_breakdown['missile_cost'] += self.config.missile_fired

        # 检测导弹命中（通过敌方is_fired_num变化或其他方式）
        # 这里需要根据实际仿真返回的数据结构调整
        if 'hit_events' in curr_obs:
            for hit in curr_obs['hit_events']:
                reward += self.config.missile_hit
                self.state.total_missiles_hit += 1
                self.reward_breakdown['missile_hit'] += self.config.missile_hit
                print(f"  [REWARD] 导弹命中 +{self.config.missile_hit}")

        return reward

    def _check_crashes(self, prev_obs: Dict, curr_obs: Dict) -> float:
        """检测非战斗坠机（高度过低、失控等）"""
        reward = 0.0

        # 检测高度过低
        if 'platform_list' in curr_obs:
            for p in curr_obs['platform_list']:
                alt = p.get('Alt', p.get('alt', p.get('Z', 1000)))
                if alt < 100:  # 高度阈值
                    # 可能即将坠机，给予警告惩罚
                    reward -= 0.1

        # 检测坠机事件
        if 'crash_events' in curr_obs:
            for crash in curr_obs['crash_events']:
                pid = crash.get('ID', crash.get('id', 0))
                # 排除战斗损失，只计算非战斗坠机
                if pid not in self._get_combat_losses(prev_obs, curr_obs):
                    reward += self.config.crash_penalty
                    self.state.own_crashed += 1
                    self.reward_breakdown['crash'] += self.config.crash_penalty
                    print(f"  [REWARD] 非战斗坠机 {self.config.crash_penalty}")

        return reward

    def _get_combat_losses(self, prev_obs: Dict, curr_obs: Dict) -> Set[int]:
        """获取战斗损失的单位ID"""
        combat_losses = set()
        if 'broken_list' in curr_obs:
            for b in curr_obs['broken_list']:
                combat_losses.add(b.get('ID', b.get('id', 0)))
        return combat_losses

    def _compute_auxiliary_rewards(self, curr_obs: Dict) -> float:
        """计算辅助奖励"""
        reward = 0.0

        # 有人机存活奖励
        manned_count = len(self.state.own_manned_alive)
        reward += manned_count * self.config.protect_manned_bonus

        # 存活时间奖励
        reward += self.config.survival_bonus

        self.reward_breakdown['auxiliary'] += reward
        return reward

    def finalize(self, victory: bool) -> float:
        """
        对战结束，计算终局奖励

        Args:
            victory: 是否胜利

        Returns:
            终局奖励
        """
        reward = 0.0

        if victory:
            reward += self.config.victory_bonus
            self.reward_breakdown['victory'] = self.config.victory_bonus
            print(f"  [REWARD] 胜利 +{self.config.victory_bonus}")
        else:
            reward += self.config.defeat_penalty
            self.reward_breakdown['victory'] = self.config.defeat_penalty
            print(f"  [REWARD] 失败 {self.config.defeat_penalty}")

        self.total_reward += reward
        return reward

    def get_summary(self) -> Dict:
        """获取奖励统计摘要"""
        return {
            'total_reward': self.total_reward,
            'breakdown': self.reward_breakdown.copy(),
            'stats': {
                'enemy_manned_killed': self.state.enemy_manned_killed,
                'enemy_uav_killed': self.state.enemy_uav_killed,
                'own_manned_lost': self.state.own_manned_lost,
                'own_uav_lost': self.state.own_uav_lost,
                'missiles_fired': self.state.total_missiles_fired,
                'missiles_hit': self.state.total_missiles_hit,
                'crashes': self.state.own_crashed,
                'steps': self.step_count
            }
        }

    def print_summary(self):
        """打印奖励统计"""
        summary = self.get_summary()
        print("\n" + "=" * 50)
        print("奖励统计摘要")
        print("=" * 50)
        print(f"总奖励: {summary['total_reward']:.2f}")
        print("\n奖励分解:")
        for key, value in summary['breakdown'].items():
            if value != 0:
                print(f"  {key}: {value:+.2f}")
        print("\n战斗统计:")
        for key, value in summary['stats'].items():
            print(f"  {key}: {value}")
        print("=" * 50)


# === 简化版奖励函数（用于快速集成）===

def compute_step_reward(prev_obs: Dict, curr_obs: Dict,
                        actions: Dict = None,
                        config: RewardConfig = None) -> float:
    """
    简化版单步奖励计算

    可直接在battle_func中使用
    """
    if config is None:
        config = RewardConfig()

    reward = 0.0

    # 检测敌方损失
    prev_enemy = set()
    curr_enemy = set()

    for t in prev_obs.get('track_list', []):
        prev_enemy.add(t.get('ID', t.get('id', 0)))
    for t in curr_obs.get('track_list', []):
        curr_enemy.add(t.get('ID', t.get('id', 0)))

    killed_enemy = prev_enemy - curr_enemy
    for _ in killed_enemy:
        # 简化：统一给击杀奖励
        reward += config.kill_enemy_uav

    # 检测己方损失
    prev_own = set()
    curr_own = set()

    for p in prev_obs.get('platform_list', []):
        prev_own.add(p.get('ID', p.get('id', 0)))
    for p in curr_obs.get('platform_list', []):
        curr_own.add(p.get('ID', p.get('id', 0)))

    lost_own = prev_own - curr_own
    for _ in lost_own:
        reward += config.lose_own_uav

    return reward


def compute_final_reward(own_manned_alive: int, enemy_manned_alive: int,
                         own_uav_alive: int, enemy_uav_alive: int,
                         config: RewardConfig = None) -> Tuple[float, bool]:
    """
    计算终局奖励

    Returns:
        (reward, victory)
    """
    if config is None:
        config = RewardConfig()

    # 胜利条件：敌方有人机全灭
    victory = enemy_manned_alive == 0 and own_manned_alive > 0

    if victory:
        return config.victory_bonus, True
    elif own_manned_alive == 0:
        return config.defeat_penalty, False
    else:
        # 未分胜负，按存活比例给分
        own_score = own_manned_alive * 3 + own_uav_alive
        enemy_score = enemy_manned_alive * 3 + enemy_uav_alive
        ratio = (own_score - enemy_score) / max(own_score + enemy_score, 1)
        return ratio * 20, ratio > 0


# 测试代码
if __name__ == "__main__":
    # 测试奖励计算器
    calc = RewardCalculator()

    # 模拟初始状态
    initial_obs = {
        'platform_list': [
            {'ID': 1, 'Type': 1},  # 有人机
            {'ID': 2, 'Type': 2},  # 无人机
            {'ID': 3, 'Type': 2},
            {'ID': 4, 'Type': 2},
            {'ID': 5, 'Type': 2},
        ],
        'track_list': [
            {'ID': 101, 'Type': 1},  # 敌方有人机
            {'ID': 102, 'Type': 2},  # 敌方无人机
            {'ID': 103, 'Type': 2},
            {'ID': 104, 'Type': 2},
            {'ID': 105, 'Type': 2},
        ]
    }

    calc.reset(initial_obs)

    # 模拟击毁敌方无人机
    prev_obs = initial_obs.copy()
    curr_obs = {
        'platform_list': initial_obs['platform_list'],
        'track_list': [
            {'ID': 101, 'Type': 1},
            {'ID': 102, 'Type': 2},
            {'ID': 103, 'Type': 2},
            # ID 104, 105 被击毁
        ]
    }

    reward = calc.step(prev_obs, curr_obs)
    print(f"Step reward: {reward}")

    # 结束对战
    calc.finalize(victory=True)
    calc.print_summary()
