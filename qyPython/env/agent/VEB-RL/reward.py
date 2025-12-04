"""
奖励函数模块

设计原则:
1. 有人机价值 > 无人机价值（有人机是胜负关键）
2. 击毁敌方 > 保护己方（进攻性策略）
3. 导弹精准使用，避免浪费
4. 坠机惩罚，避免自杀式策略
5. 战术配合奖励，鼓励协同作战

数据结构说明（基于仿真环境）:
- platform_list: 己方平台列表
  - type: "有人机" / "无人机"
  - name: 平台名称（如"红有人机1"）
  - weapons[].quantity: 剩余导弹数
  - altitude: 高度（米）
  - longitude/latitude: 经纬度
- track_list: 探测到的目标列表
  - target_id: 目标ID
  - target_name: 目标名称
  - platform_entity_type: "有人机" / "无人机" / "导弹"
  - platform_entity_side: "red" / "blue"
  - is_fired_num: 该目标被己方导弹攻击数量
- broken_list: 已损失单位名称列表（字符串列表）
"""
import numpy as np
import math
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field


@dataclass
class RewardConfig:
    """奖励函数配置"""

    # === 击毁奖励 ===
    kill_enemy_manned: float = 50.0      # 击毁敌方有人机（核心目标）
    kill_enemy_uav: float = 15.0         # 击毁敌方无人机

    # === 损失惩罚 ===
    lose_own_manned: float = -40.0       # 己方有人机被击落（严重惩罚）
    lose_own_uav: float = -10.0          # 己方无人机被击落

    # === 导弹相关 ===
    missile_fired: float = -0.5          # 发射导弹成本（轻微成本）
    missile_hit: float = 8.0             # 导弹命中额外奖励
    missile_miss: float = -3.0           # 导弹未命中惩罚（浪费弹药）
    missile_wasted: float = -5.0         # 导弹被拦截/失效

    # === 坠机惩罚 ===
    crash_penalty: float = -15.0         # 非战斗坠机惩罚
    low_altitude_warning: float = -0.5   # 低高度警告（每步）

    # === 胜负奖励 ===
    victory_bonus: float = 100.0         # 胜利奖励（大幅提升）
    defeat_penalty: float = -50.0        # 失败惩罚

    # === 战术奖励 ===
    protect_manned_bonus: float = 0.3    # 每步有人机存活奖励
    lock_enemy_manned: float = 2.0       # 锁定敌方有人机奖励（导弹锁定）
    coordinated_attack: float = 3.0      # 多弹协同攻击奖励（is_fired_num > 1）
    survival_bonus: float = 0.02         # 存活时间奖励（每步）

    # === 侦查发现奖励（首次发现敌方单位）===
    discover_enemy_manned: float = 8.0   # 首次发现敌方有人机（高价值目标）
    discover_enemy_uav: float = 3.0      # 首次发现敌方无人机
    maintain_tracking: float = 0.1       # 持续跟踪敌方有人机奖励（每步）

    # === 导弹躲避奖励 ===
    evade_missile_manned: float = 10.0   # 有人机成功躲避导弹（高价值）
    evade_missile_uav: float = 3.0       # 无人机成功躲避导弹
    under_missile_threat: float = -0.2   # 被导弹锁定时的威胁惩罚（每步）

    # === 距离相关奖励 ===
    approach_enemy_manned: float = 0.1   # 接近敌方有人机奖励（每km）
    keep_safe_distance: float = 0.05     # 有人机保持安全距离奖励
    uav_screen_bonus: float = 0.2        # 无人机在有人机前方掩护奖励

    # === 阈值配置 ===
    low_altitude_threshold: float = 500.0      # 低高度警告阈值（米）
    safe_distance_min: float = 5000.0          # 有人机最小安全距离（米）
    safe_distance_max: float = 15000.0         # 有人机最大安全距离（米）

    # === 边界区域配置（200km×200km）===
    # 边界中心坐标（需根据实际地图调整）
    boundary_center_longitude: float = 146.1   # 边界中心经度
    boundary_center_latitude: float = 33.3     # 边界中心纬度
    boundary_half_size: float = 100000.0       # 边界半边长（米），200km/2=100km

    # === 边界违规惩罚 ===
    boundary_warning_distance: float = 10000.0 # 接近边界警告距离（10km）
    boundary_warning_penalty: float = -1.0     # 接近边界警告惩罚（每步）
    uav_boundary_violation: float = -20.0      # 无人机越界惩罚（立即销毁）
    manned_boundary_violation: float = -5.0    # 有人机越界惩罚（每步，30秒内需返回）
    manned_boundary_timeout: float = 30.0      # 有人机越界宽限时间（秒）
    manned_boundary_destroy: float = -100.0    # 有人机越界超时销毁惩罚（=判负）

    # === 中心区域占据奖励（僵持阶段关键）===
    center_longitude: float = 146.1            # 中心区域经度（需根据实际地图调整）
    center_latitude: float = 33.3              # 中心区域纬度（需根据实际地图调整）
    center_radius: float = 20000.0             # 中心区域半径（米）
    center_occupy_bonus: float = 1.0           # 普通单位在中心区域奖励（每步）
    center_manned_bonus: float = 2.0           # 有人机在中心区域额外奖励（每步）
    center_control_bonus: float = 5.0          # 己方单位数量优势时的控制奖励

    # === 僵持阶段配置 ===
    stalemate_threshold_steps: int = 300       # 判定僵持的步数阈值
    stalemate_center_weight: float = 2.0       # 僵持阶段中心区域奖励权重倍数
    no_missiles_bonus_multiplier: float = 3.0  # 双方导弹耗尽时中心奖励倍数


@dataclass
class BattleState:
    """战场状态跟踪"""

    # 存活单位名称集合
    own_manned_alive: Set[str] = field(default_factory=set)
    own_uav_alive: Set[str] = field(default_factory=set)
    enemy_manned_alive: Set[str] = field(default_factory=set)
    enemy_uav_alive: Set[str] = field(default_factory=set)

    # 己方发射的导弹（跟踪中）
    own_missiles_in_flight: Set[str] = field(default_factory=set)

    # 上一步的broken_list（用于检测新增损失）
    prev_broken_list: Set[str] = field(default_factory=set)

    # === 侦查发现跟踪 ===
    discovered_enemy_manned: Set[str] = field(default_factory=set)  # 已发现的敌方有人机
    discovered_enemy_uav: Set[str] = field(default_factory=set)     # 已发现的敌方无人机
    currently_tracked_enemies: Set[str] = field(default_factory=set)  # 当前正在跟踪的敌方单位

    # === 敌方来袭导弹跟踪 ===
    enemy_missiles_in_flight: Set[str] = field(default_factory=set)  # 当前飞行中的敌方导弹
    enemy_missiles_targeting: Dict[str, str] = field(default_factory=dict)  # 导弹名 -> 目标己方单位名

    # 统计
    total_missiles_fired: int = 0
    total_missiles_hit: int = 0
    total_missiles_miss: int = 0

    # 击杀统计
    enemy_manned_killed: int = 0
    enemy_uav_killed: int = 0
    own_manned_lost: int = 0
    own_uav_lost: int = 0
    own_crashed: int = 0

    # 侦查统计
    total_enemy_manned_discovered: int = 0  # 累计发现敌方有人机数量
    total_enemy_uav_discovered: int = 0     # 累计发现敌方无人机数量

    # 导弹躲避统计
    missiles_evaded_manned: int = 0         # 有人机躲避导弹次数
    missiles_evaded_uav: int = 0            # 无人机躲避导弹次数

    # === 边界违规跟踪 ===
    manned_outside_boundary: Dict[str, float] = field(default_factory=dict)  # 有人机名 -> 越界开始时间(秒)
    uav_destroyed_by_boundary: int = 0      # 因越界被销毁的无人机数量
    manned_destroyed_by_boundary: int = 0   # 因越界超时被销毁的有人机数量

    # 中心区域占据统计
    own_center_occupy_steps: int = 0       # 己方在中心区域的累计步数
    enemy_center_occupy_steps: int = 0     # 敌方在中心区域的累计步数
    is_stalemate: bool = False             # 是否进入僵持阶段
    no_missiles_detected: bool = False     # 是否检测到双方导弹耗尽


class RewardCalculator:
    """
    奖励计算器

    使用方法:
    1. 每场对战开始时调用 reset(initial_obs)
    2. 每步调用 step(prev_obs, curr_obs, actions) 获取即时奖励
    3. 对战结束时调用 finalize(victory) 获取终局奖励
    """

    def __init__(self, config: Optional[RewardConfig] = None, side: str = "red"):
        self.config = config or RewardConfig()
        self.side = side  # "red" 或 "blue"
        self.enemy_side = "blue" if side == "red" else "red"
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
            'tactical': 0.0,
            'auxiliary': 0.0,
            'center_control': 0.0,  # 中心区域控制奖励
            'stalemate_bonus': 0.0, # 僵持阶段奖励
            'reconnaissance': 0.0,  # 侦查发现奖励
            'evasion': 0.0,         # 导弹躲避奖励
            'boundary': 0.0         # 边界违规惩罚
        }

        if initial_obs:
            self._parse_initial_state(initial_obs)

    def _parse_initial_state(self, obs: Dict):
        """解析初始战场状态"""
        # 解析己方单位
        for platform in obs.get('platform_list', []):
            name = platform.get('name', '')
            ptype = platform.get('type', '')

            if ptype == '有人机':
                self.state.own_manned_alive.add(name)
            elif ptype == '无人机':
                self.state.own_uav_alive.add(name)

        # 解析敌方单位（从track_list中）
        for track in obs.get('track_list', []):
            name = track.get('target_name', '')
            entity_type = track.get('platform_entity_type', '')
            entity_side = track.get('platform_entity_side', '')

            # 只统计敌方飞机，不统计导弹
            if entity_side == self.enemy_side and entity_type != '导弹':
                if entity_type == '有人机':
                    self.state.enemy_manned_alive.add(name)
                elif entity_type == '无人机':
                    self.state.enemy_uav_alive.add(name)

        # 初始化broken_list
        self.state.prev_broken_list = set(obs.get('broken_list', []))

    def step(self, prev_obs: Dict, curr_obs: Dict, actions: Dict = None) -> float:
        """
        计算单步奖励

        Args:
            prev_obs: 上一步观测（side_data格式）
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

        # 3. 检测导弹状态
        reward += self._check_missiles(prev_obs, curr_obs, actions)

        # 4. 检测低高度警告
        reward += self._check_altitude(curr_obs)

        # 5. 检测导弹躲避（敌方导弹消失但己方单位存活）
        reward += self._check_missile_evasion(prev_obs, curr_obs)

        # 6. 侦查发现奖励（在战术奖励之前，因为发现是战斗的前提）
        reward += self._compute_reconnaissance_rewards(curr_obs)

        # 7. 战术奖励
        reward += self._compute_tactical_rewards(curr_obs)

        # 8. 辅助奖励
        reward += self._compute_auxiliary_rewards(curr_obs)

        # 9. 中心区域占据奖励
        reward += self._compute_center_control_reward(curr_obs)

        # 10. 检测僵持阶段
        self._check_stalemate(curr_obs)

        # 11. 边界违规检测（无人机越界立即销毁，有人机30秒宽限）
        sim_time = curr_obs.get('sim_time', None)  # 尝试获取仿真时间
        reward += self._check_boundary_violations(curr_obs, sim_time)

        # 更新prev_broken_list
        self.state.prev_broken_list = set(curr_obs.get('broken_list', []))

        self.total_reward += reward
        return reward

    def _check_enemy_kills(self, prev_obs: Dict, curr_obs: Dict) -> float:
        """检测敌方单位被击毁（通过broken_list变化）"""
        reward = 0.0

        curr_broken = set(curr_obs.get('broken_list', []))
        new_broken = curr_broken - self.state.prev_broken_list

        for broken_name in new_broken:
            # 检查是否是敌方单位（名称包含敌方标识）
            # 例如："蓝有人机1", "蓝无人机2"
            if self.enemy_side == "blue" and broken_name.startswith("蓝"):
                if "有人机" in broken_name and "_导弹" not in broken_name:
                    reward += self.config.kill_enemy_manned
                    self.state.enemy_manned_killed += 1
                    self.state.enemy_manned_alive.discard(broken_name)
                    self.reward_breakdown['kill_manned'] += self.config.kill_enemy_manned
                    print(f"  [+{self.config.kill_enemy_manned}] 击毁敌方有人机: {broken_name}")
                elif "无人机" in broken_name and "_导弹" not in broken_name:
                    reward += self.config.kill_enemy_uav
                    self.state.enemy_uav_killed += 1
                    self.state.enemy_uav_alive.discard(broken_name)
                    self.reward_breakdown['kill_uav'] += self.config.kill_enemy_uav
                    print(f"  [+{self.config.kill_enemy_uav}] 击毁敌方无人机: {broken_name}")

            elif self.enemy_side == "red" and broken_name.startswith("红"):
                if "有人机" in broken_name and "_导弹" not in broken_name:
                    reward += self.config.kill_enemy_manned
                    self.state.enemy_manned_killed += 1
                    self.state.enemy_manned_alive.discard(broken_name)
                    self.reward_breakdown['kill_manned'] += self.config.kill_enemy_manned
                elif "无人机" in broken_name and "_导弹" not in broken_name:
                    reward += self.config.kill_enemy_uav
                    self.state.enemy_uav_killed += 1
                    self.state.enemy_uav_alive.discard(broken_name)
                    self.reward_breakdown['kill_uav'] += self.config.kill_enemy_uav

        return reward

    def _check_own_losses(self, prev_obs: Dict, curr_obs: Dict) -> float:
        """检测己方损失"""
        reward = 0.0

        curr_broken = set(curr_obs.get('broken_list', []))
        new_broken = curr_broken - self.state.prev_broken_list

        side_prefix = "红" if self.side == "red" else "蓝"

        for broken_name in new_broken:
            if broken_name.startswith(side_prefix):
                if "有人机" in broken_name and "_导弹" not in broken_name:
                    reward += self.config.lose_own_manned
                    self.state.own_manned_lost += 1
                    self.state.own_manned_alive.discard(broken_name)
                    self.reward_breakdown['lose_manned'] += self.config.lose_own_manned
                    print(f"  [{self.config.lose_own_manned}] 己方有人机被击落: {broken_name}")
                elif "无人机" in broken_name and "_导弹" not in broken_name:
                    reward += self.config.lose_own_uav
                    self.state.own_uav_lost += 1
                    self.state.own_uav_alive.discard(broken_name)
                    self.reward_breakdown['lose_uav'] += self.config.lose_own_uav
                    print(f"  [{self.config.lose_own_uav}] 己方无人机被击落: {broken_name}")
                elif "_导弹" in broken_name:
                    # 己方导弹消失，可能命中或未命中
                    # 需要结合其他信息判断
                    self.state.own_missiles_in_flight.discard(broken_name)

        return reward

    def _check_missiles(self, prev_obs: Dict, curr_obs: Dict, actions: Dict) -> float:
        """检测导弹发射和命中"""
        reward = 0.0

        # 检测新发射的己方导弹（通过track_list中的己方导弹）
        for track in curr_obs.get('track_list', []):
            name = track.get('target_name', '')
            entity_type = track.get('platform_entity_type', '')
            entity_side = track.get('platform_entity_side', '')

            if entity_type == '导弹' and entity_side == self.side:
                if name not in self.state.own_missiles_in_flight:
                    # 新发射的导弹
                    self.state.own_missiles_in_flight.add(name)
                    self.state.total_missiles_fired += 1
                    reward += self.config.missile_fired
                    self.reward_breakdown['missile_cost'] += self.config.missile_fired

        # 检测协同攻击奖励（敌方单位被多枚导弹锁定）
        for track in curr_obs.get('track_list', []):
            entity_side = track.get('platform_entity_side', '')
            entity_type = track.get('platform_entity_type', '')
            is_fired_num = track.get('is_fired_num', 0)

            if entity_side == self.enemy_side and entity_type in ['有人机', '无人机']:
                if is_fired_num > 1:
                    # 协同攻击奖励
                    coord_bonus = self.config.coordinated_attack * (is_fired_num - 1)
                    reward += coord_bonus
                    self.reward_breakdown['tactical'] += coord_bonus

        return reward

    def _check_altitude(self, curr_obs: Dict) -> float:
        """检测低高度警告"""
        reward = 0.0

        for platform in curr_obs.get('platform_list', []):
            alt = platform.get('altitude', 3000)
            if alt < self.config.low_altitude_threshold:
                reward += self.config.low_altitude_warning
                self.reward_breakdown['crash'] += self.config.low_altitude_warning

        return reward

    def _check_missile_evasion(self, prev_obs: Dict, curr_obs: Dict) -> float:
        """
        检测导弹躲避奖励

        逻辑：
        1. 跟踪敌方来袭导弹（track_list中 side=enemy, type=导弹）
        2. 根据导弹名称推断目标（如"蓝有人机1_导弹_1"目标可能是红方单位）
        3. 当敌方导弹消失（进入broken_list）但对应己方单位仍存活，则判定为躲避成功
        4. 如果己方单位被导弹锁定（导弹正在飞行中），给予威胁惩罚
        """
        reward = 0.0

        # 获取当前己方存活单位
        own_alive = set()
        own_manned_names = set()
        for platform in curr_obs.get('platform_list', []):
            name = platform.get('name', '')
            own_alive.add(name)
            if platform.get('type') == '有人机':
                own_manned_names.add(name)

        # 检测当前敌方来袭导弹
        current_enemy_missiles = set()
        for track in curr_obs.get('track_list', []):
            name = track.get('target_name', '')
            entity_type = track.get('platform_entity_type', '')
            entity_side = track.get('platform_entity_side', '')

            if entity_side == self.enemy_side and entity_type == '导弹':
                current_enemy_missiles.add(name)

                # 新发现的敌方导弹
                if name not in self.state.enemy_missiles_in_flight:
                    self.state.enemy_missiles_in_flight.add(name)
                    # 尝试推断目标（根据导弹命名规则：如"蓝有人机1_导弹_1"）
                    # 通常导弹会攻击最近的敌方单位，这里简化处理

        # 检测导弹威胁惩罚（有敌方导弹在飞行中）
        if current_enemy_missiles:
            threat_penalty = len(current_enemy_missiles) * self.config.under_missile_threat
            reward += threat_penalty
            self.reward_breakdown['evasion'] += threat_penalty

        # 检测消失的敌方导弹（可能命中或被躲避）
        curr_broken = set(curr_obs.get('broken_list', []))
        prev_broken = self.state.prev_broken_list

        # 检查之前飞行中的敌方导弹是否消失
        for missile_name in list(self.state.enemy_missiles_in_flight):
            # 导弹不在当前track_list中且出现在broken_list中
            if missile_name not in current_enemy_missiles:
                if missile_name in curr_broken or missile_name in prev_broken:
                    # 导弹已消失，检查己方单位是否有新损失
                    new_broken = curr_broken - prev_broken

                    # 检查是否有己方单位在这一步被击落
                    own_unit_hit = False
                    side_prefix = "红" if self.side == "red" else "蓝"
                    for broken_name in new_broken:
                        if broken_name.startswith(side_prefix) and "_导弹" not in broken_name:
                            own_unit_hit = True
                            break

                    if not own_unit_hit:
                        # 己方无损失 = 导弹被躲避！
                        # 尝试判断躲避的是有人机还是无人机
                        # 简化：根据导弹发射方推断
                        if "有人机" in missile_name:
                            # 可能是攻击己方重要目标的导弹被躲避
                            evade_reward = self.config.evade_missile_manned
                            self.state.missiles_evaded_manned += 1
                            print(f"  [+{evade_reward}] 成功躲避敌方导弹: {missile_name}")
                        else:
                            evade_reward = self.config.evade_missile_uav
                            self.state.missiles_evaded_uav += 1
                            print(f"  [+{evade_reward}] 成功躲避敌方导弹: {missile_name}")

                        reward += evade_reward
                        self.reward_breakdown['evasion'] += evade_reward

                    # 从跟踪列表中移除
                    self.state.enemy_missiles_in_flight.discard(missile_name)

        return reward

    def _compute_reconnaissance_rewards(self, curr_obs: Dict) -> float:
        """
        计算侦查发现奖励

        敌方位置未知，必须靠己方飞机侦查发现。
        首次发现敌方单位（出现在track_list中）给予奖励。
        持续跟踪敌方有人机也给予小额奖励。
        """
        reward = 0.0

        # 当前跟踪的敌方单位
        current_enemies = set()

        for track in curr_obs.get('track_list', []):
            name = track.get('target_name', '')
            entity_type = track.get('platform_entity_type', '')
            entity_side = track.get('platform_entity_side', '')

            # 只处理敌方飞机，不处理导弹
            if entity_side != self.enemy_side or entity_type == '导弹':
                continue

            current_enemies.add(name)

            # 首次发现奖励
            if entity_type == '有人机':
                if name not in self.state.discovered_enemy_manned:
                    # 首次发现敌方有人机！
                    self.state.discovered_enemy_manned.add(name)
                    self.state.total_enemy_manned_discovered += 1
                    reward += self.config.discover_enemy_manned
                    self.reward_breakdown['reconnaissance'] += self.config.discover_enemy_manned
                    print(f"  [+{self.config.discover_enemy_manned}] 发现敌方有人机: {name}")

            elif entity_type == '无人机':
                if name not in self.state.discovered_enemy_uav:
                    # 首次发现敌方无人机
                    self.state.discovered_enemy_uav.add(name)
                    self.state.total_enemy_uav_discovered += 1
                    reward += self.config.discover_enemy_uav
                    self.reward_breakdown['reconnaissance'] += self.config.discover_enemy_uav
                    print(f"  [+{self.config.discover_enemy_uav}] 发现敌方无人机: {name}")

        # 持续跟踪敌方有人机奖励
        tracked_manned = current_enemies & self.state.discovered_enemy_manned
        if tracked_manned:
            tracking_reward = len(tracked_manned) * self.config.maintain_tracking
            reward += tracking_reward
            self.reward_breakdown['reconnaissance'] += tracking_reward

        # 更新当前跟踪状态
        self.state.currently_tracked_enemies = current_enemies

        return reward

    def _compute_tactical_rewards(self, curr_obs: Dict) -> float:
        """计算战术奖励"""
        reward = 0.0

        # 锁定敌方有人机奖励
        for track in curr_obs.get('track_list', []):
            entity_side = track.get('platform_entity_side', '')
            entity_type = track.get('platform_entity_type', '')
            is_fired_num = track.get('is_fired_num', 0)

            if entity_side == self.enemy_side and entity_type == '有人机':
                if is_fired_num > 0:
                    reward += self.config.lock_enemy_manned
                    self.reward_breakdown['tactical'] += self.config.lock_enemy_manned

        return reward

    def _compute_auxiliary_rewards(self, curr_obs: Dict) -> float:
        """计算辅助奖励"""
        reward = 0.0

        # 有人机存活奖励
        manned_count = sum(1 for p in curr_obs.get('platform_list', [])
                          if p.get('type') == '有人机')
        reward += manned_count * self.config.protect_manned_bonus

        # 存活时间奖励
        reward += self.config.survival_bonus

        self.reward_breakdown['auxiliary'] += reward
        return reward

    def _compute_center_control_reward(self, curr_obs: Dict) -> float:
        """计算中心区域控制奖励"""
        reward = 0.0

        # 统计己方和敌方在中心区域的单位数量
        own_in_center = 0
        own_manned_in_center = 0
        enemy_in_center = 0

        center_lon = self.config.center_longitude
        center_lat = self.config.center_latitude
        center_radius = self.config.center_radius

        # 检查己方单位
        for platform in curr_obs.get('platform_list', []):
            lon = platform.get('longitude', platform.get('X', 0))
            lat = platform.get('latitude', platform.get('Y', 0))
            dist = compute_distance(lon, lat, center_lon, center_lat)

            if dist <= center_radius:
                own_in_center += 1
                if platform.get('type') == '有人机':
                    own_manned_in_center += 1

        # 检查敌方单位（从track_list）
        for track in curr_obs.get('track_list', []):
            entity_side = track.get('platform_entity_side', '')
            entity_type = track.get('platform_entity_type', '')

            if entity_side == self.enemy_side and entity_type in ['有人机', '无人机']:
                lon = track.get('longitude', track.get('X', 0))
                lat = track.get('latitude', track.get('Y', 0))
                dist = compute_distance(lon, lat, center_lon, center_lat)

                if dist <= center_radius:
                    enemy_in_center += 1

        # 更新占据统计
        if own_in_center > 0:
            self.state.own_center_occupy_steps += 1
        if enemy_in_center > 0:
            self.state.enemy_center_occupy_steps += 1

        # 计算奖励倍数（僵持或无导弹时增加）
        multiplier = 1.0
        if self.state.is_stalemate:
            multiplier *= self.config.stalemate_center_weight
        if self.state.no_missiles_detected:
            multiplier *= self.config.no_missiles_bonus_multiplier

        # 基础占据奖励
        reward += own_in_center * self.config.center_occupy_bonus * multiplier

        # 有人机在中心的额外奖励
        reward += own_manned_in_center * self.config.center_manned_bonus * multiplier

        # 区域控制优势奖励（己方多于敌方）
        if own_in_center > enemy_in_center:
            advantage = own_in_center - enemy_in_center
            reward += self.config.center_control_bonus * advantage * multiplier

        self.reward_breakdown['center_control'] += reward
        return reward

    def _check_stalemate(self, curr_obs: Dict):
        """检测僵持阶段"""
        # 步数达到阈值
        if self.step_count >= self.config.stalemate_threshold_steps:
            self.state.is_stalemate = True

        # 检测双方导弹是否耗尽
        own_missiles = 0
        for platform in curr_obs.get('platform_list', []):
            for weapon in platform.get('weapons', []):
                own_missiles += weapon.get('quantity', 0)

        # 如果己方导弹耗尽，标记（假设敌方也类似）
        if own_missiles == 0:
            self.state.no_missiles_detected = True

    def _check_boundary_violations(self, curr_obs: Dict, sim_time: float = None) -> float:
        """
        检测边界违规

        规则：
        1. 无人机离开200km×200km区域 -> 立即销毁，给予惩罚
        2. 有人机离开区域 -> 30秒宽限期，期间每步惩罚
        3. 有人机宽限期超时 -> 销毁，严重惩罚（=判负）

        Args:
            curr_obs: 当前观测
            sim_time: 当前仿真时间（秒），如果为None则使用step_count估算

        Returns:
            边界违规惩罚
        """
        reward = 0.0

        # 使用仿真时间或步数估算时间（假设每步0.1秒）
        current_time = sim_time if sim_time is not None else self.step_count * 0.1

        # 边界参数
        center_lon = self.config.boundary_center_longitude
        center_lat = self.config.boundary_center_latitude
        half_size = self.config.boundary_half_size
        warning_dist = self.config.boundary_warning_distance

        # 当前存活的有人机集合（用于检测返回）
        current_manned_alive = set()

        for platform in curr_obs.get('platform_list', []):
            name = platform.get('name', '')
            ptype = platform.get('type', '')
            lon = platform.get('longitude', platform.get('X', 0))
            lat = platform.get('latitude', platform.get('Y', 0))

            # 计算到边界中心的距离
            dist_to_center = compute_distance(lon, lat, center_lon, center_lat)

            # 计算到边界的距离（简化为圆形边界，实际应为方形）
            # 对于方形边界，需要分别检查经纬度
            # 这里使用简化的圆形边界估算
            dist_to_boundary = half_size - dist_to_center

            # 检查是否越界
            is_outside = dist_to_boundary < 0

            # 检查是否接近边界（警告区域）
            is_near_boundary = 0 < dist_to_boundary < warning_dist

            if ptype == '无人机':
                if is_outside:
                    # 无人机越界 = 立即销毁
                    reward += self.config.uav_boundary_violation
                    self.state.uav_destroyed_by_boundary += 1
                    self.reward_breakdown['boundary'] += self.config.uav_boundary_violation
                    print(f"  [{self.config.uav_boundary_violation}] 无人机越界销毁: {name}")
                elif is_near_boundary:
                    # 无人机接近边界警告
                    reward += self.config.boundary_warning_penalty
                    self.reward_breakdown['boundary'] += self.config.boundary_warning_penalty

            elif ptype == '有人机':
                current_manned_alive.add(name)

                if is_outside:
                    if name not in self.state.manned_outside_boundary:
                        # 有人机首次越界，记录时间
                        self.state.manned_outside_boundary[name] = current_time
                        print(f"  [警告] 有人机越界，开始计时: {name}")

                    # 计算已越界时间
                    time_outside = current_time - self.state.manned_outside_boundary[name]

                    if time_outside >= self.config.manned_boundary_timeout:
                        # 超时销毁！
                        reward += self.config.manned_boundary_destroy
                        self.state.manned_destroyed_by_boundary += 1
                        self.reward_breakdown['boundary'] += self.config.manned_boundary_destroy
                        print(f"  [{self.config.manned_boundary_destroy}] 有人机越界超时销毁: {name} (越界{time_outside:.1f}秒)")
                        # 从跟踪中移除
                        del self.state.manned_outside_boundary[name]
                    else:
                        # 越界中，给予持续惩罚
                        reward += self.config.manned_boundary_violation
                        self.reward_breakdown['boundary'] += self.config.manned_boundary_violation
                        remaining = self.config.manned_boundary_timeout - time_outside
                        print(f"  [{self.config.manned_boundary_violation}] 有人机越界中: {name} (剩余{remaining:.1f}秒)")

                else:
                    # 有人机在界内
                    if name in self.state.manned_outside_boundary:
                        # 有人机返回界内，解除警告
                        del self.state.manned_outside_boundary[name]
                        print(f"  [解除] 有人机返回界内: {name}")

                    if is_near_boundary:
                        # 有人机接近边界警告
                        reward += self.config.boundary_warning_penalty
                        self.reward_breakdown['boundary'] += self.config.boundary_warning_penalty

        # 清理已不存在的有人机跟踪记录
        for name in list(self.state.manned_outside_boundary.keys()):
            if name not in current_manned_alive:
                del self.state.manned_outside_boundary[name]

        return reward

    def _get_center_control_time_ratio(self) -> float:
        """获取中心区域控制时间比例"""
        total = self.state.own_center_occupy_steps + self.state.enemy_center_occupy_steps
        if total == 0:
            return 0.5
        return self.state.own_center_occupy_steps / total

    def finalize(self, victory: bool) -> float:
        """对战结束，计算终局奖励"""
        reward = 0.0

        if victory:
            reward += self.config.victory_bonus
            self.reward_breakdown['victory'] = self.config.victory_bonus
            print(f"  [+{self.config.victory_bonus}] 胜利!")
        else:
            reward += self.config.defeat_penalty
            self.reward_breakdown['victory'] = self.config.defeat_penalty
            print(f"  [{self.config.defeat_penalty}] 失败")

        # 僵持阶段的中心区域控制奖励
        if self.state.is_stalemate or self.state.no_missiles_detected:
            center_ratio = self._get_center_control_time_ratio()
            # 如果己方占据中心时间更长，给予额外奖励
            if center_ratio > 0.5:
                stalemate_bonus = (center_ratio - 0.5) * 100  # 最多+50
                reward += stalemate_bonus
                self.reward_breakdown['stalemate_bonus'] = stalemate_bonus
                print(f"  [+{stalemate_bonus:.1f}] 中心区域控制优势 (占据比例: {center_ratio:.1%})")
            elif center_ratio < 0.5:
                stalemate_penalty = (0.5 - center_ratio) * 60  # 最多-30
                reward -= stalemate_penalty
                self.reward_breakdown['stalemate_bonus'] = -stalemate_penalty
                print(f"  [-{stalemate_penalty:.1f}] 中心区域控制劣势 (占据比例: {center_ratio:.1%})")

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
                'steps': self.step_count,
                # 侦查统计
                'enemy_manned_discovered': self.state.total_enemy_manned_discovered,
                'enemy_uav_discovered': self.state.total_enemy_uav_discovered,
                'currently_tracking': len(self.state.currently_tracked_enemies),
                # 导弹躲避统计
                'missiles_evaded_manned': self.state.missiles_evaded_manned,
                'missiles_evaded_uav': self.state.missiles_evaded_uav,
                'enemy_missiles_in_flight': len(self.state.enemy_missiles_in_flight),
                # 中心区域统计
                'own_center_occupy_steps': self.state.own_center_occupy_steps,
                'enemy_center_occupy_steps': self.state.enemy_center_occupy_steps,
                'center_control_ratio': self._get_center_control_time_ratio(),
                'is_stalemate': self.state.is_stalemate,
                'no_missiles': self.state.no_missiles_detected,
                # 边界违规统计
                'uav_destroyed_by_boundary': self.state.uav_destroyed_by_boundary,
                'manned_destroyed_by_boundary': self.state.manned_destroyed_by_boundary,
                'manned_currently_outside': len(self.state.manned_outside_boundary)
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


# === 工具函数 ===

def compute_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    计算两点之间的距离（米）
    使用Haversine公式
    """
    R = 6371000  # 地球半径（米）
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c


def get_unit_counts(obs: Dict, side: str) -> Dict[str, int]:
    """
    统计单位数量

    Returns:
        {'own_manned': n, 'own_uav': n, 'enemy_manned': n, 'enemy_uav': n}
    """
    enemy_side = "blue" if side == "red" else "red"

    own_manned = sum(1 for p in obs.get('platform_list', []) if p.get('type') == '有人机')
    own_uav = sum(1 for p in obs.get('platform_list', []) if p.get('type') == '无人机')

    enemy_manned = sum(1 for t in obs.get('track_list', [])
                       if t.get('platform_entity_side') == enemy_side
                       and t.get('platform_entity_type') == '有人机')
    enemy_uav = sum(1 for t in obs.get('track_list', [])
                    if t.get('platform_entity_side') == enemy_side
                    and t.get('platform_entity_type') == '无人机')

    return {
        'own_manned': own_manned,
        'own_uav': own_uav,
        'enemy_manned': enemy_manned,
        'enemy_uav': enemy_uav
    }


def compute_final_reward(own_manned: int, enemy_manned: int,
                         own_uav: int, enemy_uav: int,
                         config: RewardConfig = None) -> Tuple[float, bool]:
    """
    计算终局奖励

    Returns:
        (reward, victory)
    """
    if config is None:
        config = RewardConfig()

    # 胜利条件：敌方有人机全灭
    victory = enemy_manned == 0 and own_manned > 0

    if victory:
        return config.victory_bonus, True
    elif own_manned == 0:
        return config.defeat_penalty, False
    else:
        # 未分胜负，按存活比例给分
        own_score = own_manned * 3 + own_uav
        enemy_score = enemy_manned * 3 + enemy_uav
        ratio = (own_score - enemy_score) / max(own_score + enemy_score, 1)
        return ratio * 20, ratio > 0


# 测试代码
if __name__ == "__main__":
    # 使用真实数据结构测试
    calc = RewardCalculator(side="red")

    # 模拟初始状态（基于态势数据示例）
    initial_obs = {
        'platform_list': [
            {'id': 1, 'name': '红无人机1', 'type': '无人机', 'altitude': 3459},
            {'id': 25, 'name': '红无人机4', 'type': '无人机', 'altitude': 3491},
            {'id': 73, 'name': '红有人机2', 'type': '有人机', 'altitude': 3551},
        ],
        'track_list': [
            {'target_id': 89, 'target_name': '蓝无人机2', 'platform_entity_type': '无人机',
             'platform_entity_side': 'blue', 'is_fired_num': 1},
            {'target_id': 145, 'target_name': '蓝有人机1', 'platform_entity_type': '有人机',
             'platform_entity_side': 'blue', 'is_fired_num': 0},
        ],
        'broken_list': ['红无人机2', '红无人机1_导弹_1']
    }

    calc.reset(initial_obs)

    # 模拟下一步：击毁敌方无人机
    next_obs = {
        'platform_list': initial_obs['platform_list'],
        'track_list': [
            {'target_id': 145, 'target_name': '蓝有人机1', 'platform_entity_type': '有人机',
             'platform_entity_side': 'blue', 'is_fired_num': 2},  # 被多枚导弹锁定
        ],
        'broken_list': ['红无人机2', '红无人机1_导弹_1', '蓝无人机2']  # 新增
    }

    reward = calc.step(initial_obs, next_obs)
    print(f"\nStep reward: {reward:.2f}")

    # 胜利
    calc.finalize(victory=True)
    calc.print_summary()
