"""
VEB-RL火控策略核心模块

基于 Value-Evolutionary-Based Reinforcement Learning 论文实现
核心创新:
1. 维护Q网络种群 (而非策略网络)
2. 使用负TD误差作为适应度指标 (无需环境交互)
3. 精英交互机制 (只有top-N个体与环境交互)
4. RL注入机制 (将优化后的RL个体注入种群)

适用于: 空战火控决策 (开火/不开火)
"""

import math
import os
import json
import copy
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .q_network import (
    QNetworkConfig, QNetwork, DuelingQNetwork,
    create_q_network, hard_update, soft_update
)


# ==================== 配置类 ====================

@dataclass
class VEBConfig:
    """VEB-RL配置"""
    # 网络配置
    state_dim: int = 12  # 基于真实数据重新设计的12维状态空间
    action_dim: int = 2  # 0: 不开火, 1: 开火
    hidden_dims: Tuple[int, ...] = (256, 128)
    dueling: bool = True
    noisy: bool = False

    # 种群配置
    population_size: int = 10       # 种群大小
    elite_num: int = 2              # 精英数量 (用于交互和进化)

    # 进化配置
    ea_type: str = 'GA'             # 'GA' 或 'CEM'
    mutation_prob: float = 0.3      # 变异概率
    mutation_strength: float = 0.1  # 变异强度 (高斯噪声标准差)
    crossover_prob: float = 0.5     # 交叉概率
    tournament_size: int = 3        # 锦标赛选择大小

    # CEM配置 (当ea_type='CEM'时使用)
    cem_sigma_init: float = 1.0     # CEM初始标准差

    # RL配置
    lr: float = 3e-4                # 学习率
    gamma: float = 0.99             # 折扣因子
    batch_size: int = 64            # 批大小
    buffer_capacity: int = 100000   # 经验回放容量
    target_update_freq: int = 1000  # 目标网络更新频率 (steps)
    population_target_update_freq: int = 20  # 种群目标网络更新频率 (generations)

    # 适应度评估配置
    fitness_sample_size: int = 1024  # 用于计算适应度的样本数

    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ==================== 经验回放 ====================

class PrioritizedReplayBuffer:
    """
    优先经验回放缓冲区

    优先级基于TD误差，高误差的样本被采样概率更高
    """

    def __init__(self, capacity: int, state_dim: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.beta = beta    # 重要性采样指数
        self.beta_increment = 0.001
        self.ptr = 0
        self.size = 0
        self.max_priority = 1.0

        # 存储
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.priorities = np.zeros(capacity, dtype=np.float32)

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool):
        """添加经验"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.priorities[self.ptr] = self.max_priority

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: str = 'cpu') -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """优先采样"""
        if self.size == 0:
            return None, None, None

        # 计算采样概率
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # 采样索引
        indices = np.random.choice(self.size, size=min(batch_size, self.size), p=probs, replace=False)

        # 计算重要性采样权重
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        batch = {
            'states': torch.FloatTensor(self.states[indices]).to(device),
            'actions': torch.LongTensor(self.actions[indices]).to(device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(device),
            'next_states': torch.FloatTensor(self.next_states[indices]).to(device),
            'dones': torch.FloatTensor(self.dones[indices]).to(device)
        }

        return batch, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """更新优先级"""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6
        self.max_priority = max(self.max_priority, self.priorities[:self.size].max())

    def sample_uniform(self, batch_size: int, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """均匀采样 (用于适应度评估)"""
        if self.size == 0:
            return None

        indices = np.random.randint(0, self.size, size=min(batch_size, self.size))

        return {
            'states': torch.FloatTensor(self.states[indices]).to(device),
            'actions': torch.LongTensor(self.actions[indices]).to(device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(device),
            'next_states': torch.FloatTensor(self.next_states[indices]).to(device),
            'dones': torch.FloatTensor(self.dones[indices]).to(device)
        }

    def __len__(self) -> int:
        return self.size


# ==================== VEB-RL核心类 ====================

class VEBFireControl:
    """
    VEB-RL火控系统

    基于VEB-RL论文实现的火控决策系统
    """

    def __init__(self, config: VEBConfig = None):
        self.config = config or VEBConfig()
        self.device = torch.device(self.config.device)

        # Q网络配置
        self.q_config = QNetworkConfig(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dims=self.config.hidden_dims,
            dueling=self.config.dueling,
            noisy=self.config.noisy
        )

        # ==================== 初始化种群 ====================
        # 种群: [(Q网络, 目标Q网络), ...]
        self.population = []
        for _ in range(self.config.population_size):
            q_net = create_q_network(self.q_config, self.device)
            target_net = create_q_network(self.q_config, self.device)
            hard_update(q_net, target_net)
            self.population.append((q_net, target_net))

        # ==================== 初始化RL个体 ====================
        self.rl_q_net = create_q_network(self.q_config, self.device)
        self.rl_target_net = create_q_network(self.q_config, self.device)
        hard_update(self.rl_q_net, self.rl_target_net)

        # RL优化器
        self.optimizer = optim.Adam(self.rl_q_net.parameters(), lr=self.config.lr)

        # ==================== 经验回放 ====================
        self.replay_buffer = PrioritizedReplayBuffer(
            self.config.buffer_capacity,
            self.config.state_dim
        )

        # ==================== 训练统计 ====================
        self.train_steps = 0
        self.generation = 0
        self.fitness_history = []
        self.rl_elite_rate = []  # RL被选为精英的比率

        # CEM分布参数 (如果使用CEM)
        if self.config.ea_type == 'CEM':
            self._init_cem_distribution()

    def _init_cem_distribution(self):
        """初始化CEM分布参数"""
        # 获取参数维度
        sample_params = self._flatten_params(self.population[0][0])
        self.param_dim = len(sample_params)

        # 初始化均值和标准差
        self.cem_mean = sample_params.clone()
        self.cem_std = torch.ones(self.param_dim, device=self.device) * self.config.cem_sigma_init

    def _flatten_params(self, net: nn.Module) -> torch.Tensor:
        """将网络参数展平为一维向量"""
        return torch.cat([p.data.view(-1) for p in net.parameters()])

    def _unflatten_params(self, net: nn.Module, flat_params: torch.Tensor):
        """将一维向量恢复为网络参数"""
        idx = 0
        for p in net.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[idx:idx + numel].view(p.shape))
            idx += numel

    # ==================== 状态提取 ====================

    def extract_state(self, shooter: Dict, target: Dict,
                      friendly_shooters: List[Dict] = None) -> np.ndarray:
        """
        从射手和目标信息提取状态向量

        基于真实态势数据结构设计，充分利用速度分量信息
        新增: 协同攻击相关特征

        状态空间 (15维):
        0. 归一化距离 - 射手与目标的3D距离
        1. 归一化姿态角 - 从目标视角看射手的方位(0=尾追, 180=迎头)
        2. 接近率 - 双方相互靠近的速度 (利用速度分量精确计算)
        3. 是否在NEZ内 - 不可逃逸区判断
        4. 目标横向逃逸速度 - 垂直于导弹来袭方向的速度分量 (关键!)
        5. 目标径向速度 - 沿导弹来袭方向的速度分量 (正=远离)
        6. 高度差 - 射手相对目标的高度优势
        7. 剩余导弹比例 - 射手剩余弹药
        8. 目标被攻击数 - 已有多少导弹在攻击该目标
        9. 目标是否是有人机 - 高价值目标标识
        10. 射手是否是有人机 - 影响导弹射程
        11. 目标机动强度 - 根据roll角判断是否在剧烈机动
        === 协同攻击特征 (新增) ===
        12. 友方是否有其他射手在NEZ内 - 协同攻击机会
        13. 友方最佳协同角度差 - 与友机攻击角度的差异(越大越好)
        14. 协同攻击推荐度 - 综合评估是否适合协同攻击

        物理意义:
        - 双机协同攻击时，目标难以同时躲避两个方向的导弹
        - 最佳协同: 两架飞机从相差90°以上的方向同时攻击
        """
        # ========== 获取射手信息 ==========
        s_lon = shooter.get('longitude', 0)
        s_lat = shooter.get('latitude', 0)
        s_alt = shooter.get('altitude', 5000)
        s_speed = shooter.get('speed', 300)
        # 速度分量 (东为正x, 北为正y, 上为正z)
        s_vx = shooter.get('velocity_x', 0)
        s_vy = shooter.get('velocity_y', 0)
        s_vz = shooter.get('velocity_z', 0)
        s_heading = shooter.get('heading', 0)  # 弧度
        is_manned_shooter = shooter.get('type') == '有人机'
        shooter_id = shooter.get('id', 0)

        # 剩余导弹数量
        missiles_left = 0
        max_missiles = 4 if is_manned_shooter else 2
        weapons = shooter.get('weapons', [])
        if weapons:
            missiles_left = weapons[0].get('quantity', 0)

        # ========== 获取目标信息 ==========
        t_lon = target.get('longitude', target.get('X', 0))
        t_lat = target.get('latitude', target.get('Y', 0))
        t_alt = target.get('altitude', target.get('Alt', 5000))
        t_speed = target.get('speed', 300)
        # 速度分量 (track数据用v_x/v_y/v_z)
        t_vx = target.get('v_x', target.get('velocity_x', 0))
        t_vy = target.get('v_y', target.get('velocity_y', 0))
        t_vz = target.get('v_z', target.get('velocity_z', 0))
        t_heading = target.get('heading', 0)  # 弧度
        t_roll = target.get('roll', 0)  # 横滚角，用于判断机动
        is_manned_target = target.get('platform_entity_type') == '有人机'
        is_fired_num = target.get('is_fired_num', 0)  # 已被多少导弹攻击

        # ========== 计算3D距离 ==========
        horizontal_dist = self._haversine_distance(s_lon, s_lat, t_lon, t_lat)
        alt_diff = s_alt - t_alt
        distance_3d = math.sqrt(horizontal_dist**2 + alt_diff**2)

        # ========== 计算姿态角 ==========
        aspect_angle = self._calculate_aspect_angle(
            s_lon, s_lat, s_heading, t_lon, t_lat, t_heading
        )

        # ========== 利用速度分量精确计算接近率 ==========
        # 射手到目标的单位方向向量
        dx_m = (t_lon - s_lon) * 111000 * math.cos(math.radians((s_lat + t_lat) / 2))
        dy_m = (t_lat - s_lat) * 111000
        dz_m = t_alt - s_alt
        dist_m = math.sqrt(dx_m**2 + dy_m**2 + dz_m**2)

        if dist_m > 0:
            # 单位方向向量 (从射手指向目标)
            ux, uy, uz = dx_m / dist_m, dy_m / dist_m, dz_m / dist_m

            # 相对速度 (目标速度 - 射手速度)
            rel_vx = t_vx - s_vx
            rel_vy = t_vy - s_vy
            rel_vz = t_vz - s_vz

            # 接近率 = 相对速度在连线方向的投影 (负值表示接近)
            closure_rate = -(rel_vx * ux + rel_vy * uy + rel_vz * uz)
        else:
            closure_rate = 0

        # ========== 计算目标速度正交分解 (核心!) ==========
        v_lateral, v_radial = self._decompose_target_velocity_3d(
            s_lon, s_lat, s_alt, t_lon, t_lat, t_alt, t_vx, t_vy, t_vz
        )

        # ========== 计算NEZ (基于仿真环境参数) ==========
        # 导弹速度: 1200 m/s, 飞机速度: ~400 m/s
        # 有人机导弹性能更好 (射程更远)
        nez, max_range = self._calculate_nez(
            distance_3d, aspect_angle, s_alt, t_alt,
            s_speed, t_speed, is_manned_shooter
        )
        in_nez = 1.0 if distance_3d <= nez else 0.0

        # ========== 目标机动强度 (根据roll角) ==========
        # roll角越大，说明飞机在急转弯
        maneuver_intensity = min(abs(t_roll) / 1.4, 1.0)  # 1.4 rad ≈ 80度

        # ========== 构建状态向量 ==========
        state = np.array([
            min(distance_3d / max_range, 1.0),             # 0: 归一化3D距离
            aspect_angle / 180.0,                          # 1: 归一化姿态角
            np.clip(closure_rate / 800.0, -1, 1),         # 2: 归一化接近率
            in_nez,                                        # 3: 是否在NEZ内
            np.clip(v_lateral / 400.0, 0, 1),             # 4: 目标横向逃逸速度 (关键!)
            np.clip(v_radial / 400.0, -1, 1),             # 5: 目标径向速度
            np.clip(alt_diff / 5000.0, -1, 1),            # 6: 高度差 (正=射手更高)
            missiles_left / max_missiles,                  # 7: 剩余导弹比例
            min(is_fired_num / 3.0, 1.0),                 # 8: 目标已被攻击数
            1.0 if is_manned_target else 0.0,             # 9: 目标是否有人机
            1.0 if is_manned_shooter else 0.0,            # 10: 射手是否有人机
            maneuver_intensity                             # 11: 目标机动强度
        ], dtype=np.float32)

        return state

    def _decompose_target_velocity_3d(self, s_lon: float, s_lat: float, s_alt: float,
                                       t_lon: float, t_lat: float, t_alt: float,
                                       t_vx: float, t_vy: float, t_vz: float) -> Tuple[float, float]:
        """
        利用3D速度分量直接计算目标的横向和径向速度

        这比使用航向角计算更准确!

        Args:
            s_lon, s_lat, s_alt: 射手位置
            t_lon, t_lat, t_alt: 目标位置
            t_vx, t_vy, t_vz: 目标速度分量 (东/北/上)

        Returns:
            (v_lateral, v_radial): 横向速度(逃逸能力)和径向速度(正=远离)
        """
        # 计算射手到目标的方向向量 (米)
        dx_m = (t_lon - s_lon) * 111000 * math.cos(math.radians((s_lat + t_lat) / 2))
        dy_m = (t_lat - s_lat) * 111000
        dz_m = t_alt - s_alt

        dist = math.sqrt(dx_m**2 + dy_m**2 + dz_m**2)

        if dist < 1:  # 避免除零
            return 0, 0

        # 单位方向向量 (从射手指向目标，即导弹飞行方向)
        ux, uy, uz = dx_m / dist, dy_m / dist, dz_m / dist

        # 目标速度在导弹来袭方向(反方向)的投影
        # v_radial: 正值表示目标在远离射手
        v_radial = t_vx * ux + t_vy * uy + t_vz * uz

        # 目标速度的横向分量 (垂直于连线方向)
        # 先计算目标速度的总大小
        t_speed = math.sqrt(t_vx**2 + t_vy**2 + t_vz**2)

        # 横向速度 = sqrt(总速度² - 径向速度²)
        v_lateral_sq = t_speed**2 - v_radial**2
        v_lateral = math.sqrt(max(0, v_lateral_sq))  # 确保非负

        return v_lateral, v_radial

    def _calculate_nez(self, distance: float, aspect_angle: float,
                       shooter_alt: float, target_alt: float,
                       shooter_speed: float, target_speed: float,
                       is_manned: bool) -> Tuple[float, float]:
        """
        计算NEZ (No Escape Zone) 不可逃逸区

        基于仿真环境参数:
        - 导弹速度: 1200 m/s
        - 导弹有效飞行时间: ~30-35秒
        - 导弹最大过载: ~30G (假设)

        Args:
            distance: 当前距离 (m)
            aspect_angle: 姿态角 (0=尾追, 180=迎头)
            shooter_alt: 射手高度 (m)
            target_alt: 目标高度 (m)
            shooter_speed: 射手速度 (m/s)
            target_speed: 目标速度 (m/s)
            is_manned: 射手是否是有人机 (有人机导弹性能更好)

        Returns:
            (nez_distance, max_range): NEZ距离和最大射程 (m)
        """
        # ========== 基础参数 (基于仿真数据) ==========
        MISSILE_SPEED = 1200  # m/s
        MISSILE_FLIGHT_TIME = 35  # 秒 (有效飞行时间)

        # 有人机导弹性能更好
        if is_manned:
            base_max_range = 35000   # 35km
            base_nez_tail = 12000    # 尾追NEZ 12km
            base_nez_head = 22000    # 迎头NEZ 22km
        else:
            base_max_range = 25000   # 25km
            base_nez_tail = 8000     # 尾追NEZ 8km
            base_nez_head = 16000    # 迎头NEZ 16km

        # ========== 高度修正 ==========
        # 高空空气稀薄，导弹射程增加
        # 低空空气阻力大，射程减少
        avg_alt = (shooter_alt + target_alt) / 2
        if avg_alt > 8000:
            alt_factor = 1.15  # 高空 +15%
        elif avg_alt > 5000:
            alt_factor = 1.05  # 中高空 +5%
        elif avg_alt > 3000:
            alt_factor = 1.0   # 中空 基准
        elif avg_alt > 1000:
            alt_factor = 0.9   # 低空 -10%
        else:
            alt_factor = 0.75  # 超低空 -25%

        # ========== 高度差修正 ==========
        # 从高处向下打，射程增加 (重力辅助)
        # 从低处向上打，射程减少 (重力阻碍)
        alt_diff = shooter_alt - target_alt
        if alt_diff > 2000:
            height_adv_factor = 1.1   # 大高度优势 +10%
        elif alt_diff > 500:
            height_adv_factor = 1.05  # 小高度优势 +5%
        elif alt_diff > -500:
            height_adv_factor = 1.0   # 平飞
        elif alt_diff > -2000:
            height_adv_factor = 0.95  # 小高度劣势 -5%
        else:
            height_adv_factor = 0.85  # 大高度劣势 -15%

        # ========== 速度修正 ==========
        # 射手速度快，导弹初始能量高，射程增加
        # 目标速度快，相对速度变化，影响NEZ
        speed_factor = 1.0 + (shooter_speed - 350) / 2000  # 基准350m/s

        # ========== 应用修正 ==========
        max_range = base_max_range * alt_factor * height_adv_factor * speed_factor
        nez_tail = base_nez_tail * alt_factor * height_adv_factor * speed_factor
        nez_head = base_nez_head * alt_factor * height_adv_factor * speed_factor

        # ========== 姿态角插值计算NEZ ==========
        # 0° = 尾追 (NEZ小), 180° = 迎头 (NEZ大)
        # 使用余弦插值，更符合物理特性
        aspect_rad = math.radians(aspect_angle)
        # cos(0)=1 -> nez_tail, cos(180)=-1 -> nez_head
        aspect_factor = (1 - math.cos(aspect_rad)) / 2  # 0到1
        nez = nez_tail + (nez_head - nez_tail) * aspect_factor

        # ========== 目标速度对NEZ的影响 ==========
        # 目标速度快，更容易逃脱，NEZ减小
        target_escape_factor = 1.0 - (target_speed - 300) / 1500
        target_escape_factor = max(0.7, min(1.0, target_escape_factor))
        nez *= target_escape_factor

        return nez, max_range

    def _haversine_distance(self, lon1, lat1, lon2, lat2) -> float:
        """计算两点间距离 (米)"""
        R = 6371000  # 地球半径
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def _calculate_aspect_angle(self, s_lon, s_lat, s_heading, t_lon, t_lat, t_heading) -> float:
        """计算姿态角"""
        dx = s_lon - t_lon
        dy = s_lat - t_lat
        bearing_to_shooter = math.degrees(math.atan2(dx, dy)) % 360

        target_hdg = math.degrees(t_heading) % 360 if isinstance(t_heading, float) else t_heading % 360
        aspect = abs((bearing_to_shooter - target_hdg + 180) % 360 - 180)
        return aspect

    def _calculate_closure_rate(self, s_lon, s_lat, s_speed, s_heading,
                                 t_lon, t_lat, t_speed, t_heading) -> float:
        """计算接近率"""
        dx = t_lon - s_lon
        dy = t_lat - s_lat
        bearing_to_target = math.degrees(math.atan2(dx, dy)) % 360

        shooter_hdg = math.degrees(s_heading) % 360 if isinstance(s_heading, float) else s_heading % 360
        target_hdg = math.degrees(t_heading) % 360 if isinstance(t_heading, float) else t_heading % 360

        shooter_angle = math.radians(bearing_to_target - shooter_hdg)
        shooter_closure = s_speed * math.cos(shooter_angle)

        bearing_to_shooter = (bearing_to_target + 180) % 360
        target_angle = math.radians(bearing_to_shooter - target_hdg)
        target_closure = t_speed * math.cos(target_angle)

        return shooter_closure + target_closure

    # ==================== 适应度评估 ====================

    def calculate_fitness(self, q_net: nn.Module, target_net: nn.Module) -> float:
        """
        计算适应度: 负TD误差

        f(θ, θ') = -E[(r + γ max_a' Q_θ'(s', a') - Q_θ(s, a))²]

        适应度越高 (TD误差越小)，价值函数越准确
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return float('-inf')

        batch = self.replay_buffer.sample_uniform(
            self.config.fitness_sample_size,
            self.device
        )

        if batch is None:
            return float('-inf')

        with torch.no_grad():
            # 当前Q值
            q_values = q_net(batch['states'])
            current_q = q_values.gather(1, batch['actions'])

            # 目标Q值
            next_q = target_net(batch['next_states']).max(dim=1, keepdim=True)[0]
            target_q = batch['rewards'] + self.config.gamma * (1 - batch['dones']) * next_q

            # TD误差
            td_error = (target_q - current_q).pow(2).mean()

        # 返回负TD误差作为适应度
        return -td_error.item()

    def evaluate_population(self) -> List[float]:
        """评估种群所有个体的适应度"""
        fitness_scores = []
        for q_net, target_net in self.population:
            fitness = self.calculate_fitness(q_net, target_net)
            fitness_scores.append(fitness)
        return fitness_scores

    # ==================== 进化操作 ====================

    def evolve_population(self):
        """
        进化种群

        步骤:
        1. 评估适应度
        2. 选择精英
        3. 注入RL个体
        4. 执行进化操作 (GA或CEM)
        5. 更新目标网络
        """
        self.generation += 1

        # 1. 评估适应度
        fitness_scores = self.evaluate_population()

        # 2. 计算RL个体适应度并注入
        rl_fitness = self.calculate_fitness(self.rl_q_net, self.rl_target_net)

        # 找到适应度最低的个体，用RL个体替换
        min_fitness_idx = np.argmin(fitness_scores)
        if rl_fitness > fitness_scores[min_fitness_idx]:
            # 复制RL个体到种群
            hard_update(self.rl_q_net, self.population[min_fitness_idx][0])
            hard_update(self.rl_target_net, self.population[min_fitness_idx][1])
            fitness_scores[min_fitness_idx] = rl_fitness

        # 重新排序
        sorted_indices = np.argsort(fitness_scores)[::-1]  # 降序

        # 记录RL是否是精英
        rl_is_elite = min_fitness_idx in sorted_indices[:self.config.elite_num]
        self.rl_elite_rate.append(1.0 if rl_is_elite else 0.0)

        # 3. 执行进化
        if self.config.ea_type == 'GA':
            self._evolve_ga(sorted_indices, fitness_scores)
        else:
            self._evolve_cem(sorted_indices)

        # 4. 更新种群目标网络 (每H代)
        if self.generation % self.config.population_target_update_freq == 0:
            for q_net, target_net in self.population:
                hard_update(q_net, target_net)

        # 记录适应度历史
        self.fitness_history.append({
            'generation': self.generation,
            'mean_fitness': np.mean(fitness_scores),
            'max_fitness': np.max(fitness_scores),
            'rl_fitness': rl_fitness
        })

    def _evolve_ga(self, sorted_indices: np.ndarray, fitness_scores: List[float]):
        """GA进化"""
        elite_indices = sorted_indices[:self.config.elite_num]
        non_elite_indices = sorted_indices[self.config.elite_num:]

        # 锦标赛选择 winners
        winners = []
        for _ in range(len(non_elite_indices) // 2):
            candidates = random.sample(list(sorted_indices), self.config.tournament_size)
            winner = max(candidates, key=lambda x: fitness_scores[x])
            winners.append(winner)

        # 交叉操作
        new_individuals = []
        for i in range(0, len(non_elite_indices), 2):
            if i + 1 < len(non_elite_indices):
                parent1_idx = random.choice(list(elite_indices) + winners)
                parent2_idx = random.choice(list(elite_indices) + winners)

                if random.random() < self.config.crossover_prob:
                    child1, child2 = self._crossover(
                        self.population[parent1_idx][0],
                        self.population[parent2_idx][0]
                    )
                else:
                    child1 = copy.deepcopy(self.population[parent1_idx][0])
                    child2 = copy.deepcopy(self.population[parent2_idx][0])

                new_individuals.extend([child1, child2])

        # 替换非精英个体
        for i, idx in enumerate(non_elite_indices):
            if i < len(new_individuals):
                self.population[idx] = (new_individuals[i].to(self.device),
                                         self.population[idx][1])

        # 变异操作 (非精英)
        for idx in non_elite_indices:
            if random.random() < self.config.mutation_prob:
                self._mutate(self.population[idx][0])

    def _evolve_cem(self, sorted_indices: np.ndarray):
        """CEM进化"""
        # 选择top一半
        top_half = sorted_indices[:self.config.population_size // 2]

        # 收集精英参数
        elite_params = []
        for idx in top_half:
            params = self._flatten_params(self.population[idx][0])
            elite_params.append(params)

        elite_params = torch.stack(elite_params)

        # 更新分布
        self.cem_mean = elite_params.mean(dim=0)
        self.cem_std = elite_params.std(dim=0) + 1e-6

        # 从新分布采样生成新种群 (保留精英)
        for i, idx in enumerate(sorted_indices):
            if i >= self.config.elite_num:
                # 采样新个体
                new_params = self.cem_mean + self.cem_std * torch.randn(self.param_dim, device=self.device)
                self._unflatten_params(self.population[idx][0], new_params)

    def _crossover(self, parent1: nn.Module, parent2: nn.Module) -> Tuple[nn.Module, nn.Module]:
        """K点交叉操作"""
        child1 = create_q_network(self.q_config, 'cpu')
        child2 = create_q_network(self.q_config, 'cpu')

        params1 = self._flatten_params(parent1).cpu()
        params2 = self._flatten_params(parent2).cpu()

        # 随机选择交叉点
        crossover_point = random.randint(0, len(params1))

        child1_params = torch.cat([params1[:crossover_point], params2[crossover_point:]])
        child2_params = torch.cat([params2[:crossover_point], params1[crossover_point:]])

        self._unflatten_params(child1, child1_params)
        self._unflatten_params(child2, child2_params)

        return child1, child2

    def _mutate(self, net: nn.Module):
        """高斯变异操作"""
        with torch.no_grad():
            for param in net.parameters():
                noise = torch.randn_like(param) * self.config.mutation_strength
                param.add_(noise)

    # ==================== RL优化 ====================

    def update_rl(self) -> Dict[str, float]:
        """
        DQN更新步骤

        Returns:
            训练统计信息
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return {}

        self.train_steps += 1

        # 采样
        batch, indices, weights = self.replay_buffer.sample(
            self.config.batch_size, self.device
        )

        if batch is None:
            return {}

        weights = torch.FloatTensor(weights).to(self.device)

        # 当前Q值
        q_values = self.rl_q_net(batch['states'])
        current_q = q_values.gather(1, batch['actions'])

        # Double DQN: 用在线网络选动作，用目标网络评估
        with torch.no_grad():
            next_actions = self.rl_q_net(batch['next_states']).argmax(dim=1, keepdim=True)
            next_q = self.rl_target_net(batch['next_states']).gather(1, next_actions)
            target_q = batch['rewards'] + self.config.gamma * (1 - batch['dones']) * next_q

        # TD误差
        td_errors = (target_q - current_q).detach().cpu().numpy().flatten()

        # 加权损失
        loss = (weights * (target_q - current_q).pow(2)).mean()

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rl_q_net.parameters(), 10.0)
        self.optimizer.step()

        # 更新优先级
        self.replay_buffer.update_priorities(indices, td_errors)

        # 更新目标网络
        if self.train_steps % self.config.target_update_freq == 0:
            hard_update(self.rl_q_net, self.rl_target_net)

        return {
            'loss': loss.item(),
            'q_value': current_q.mean().item(),
            'td_error': np.abs(td_errors).mean()
        }

    # ==================== 推理接口 ====================

    def should_fire(self, shooter: Dict, target: Dict,
                    use_elite: bool = True) -> Tuple[bool, float, str]:
        """
        判断是否应该开火

        Args:
            shooter: 射手信息
            target: 目标信息
            use_elite: 是否使用精英个体 (否则用RL个体)

        Returns:
            (should_fire, confidence, reason)
        """
        state = self.extract_state(shooter, target)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if use_elite and len(self.fitness_history) > 0:
            # 使用最佳精英个体
            fitness_scores = self.evaluate_population()
            best_idx = np.argmax(fitness_scores)
            q_net = self.population[best_idx][0]
        else:
            # 使用RL个体
            q_net = self.rl_q_net

        with torch.no_grad():
            q_values = q_net(state_tensor)
            action = q_values.argmax(dim=1).item()
            q_fire = q_values[0, 1].item()  # 开火的Q值
            q_hold = q_values[0, 0].item()  # 不开火的Q值

        # 计算置信度 (softmax)
        confidence = torch.softmax(q_values, dim=1)[0, 1].item()

        should_fire = action == 1

        if should_fire:
            reason = f"VEB开火(Q={q_fire:.2f}, conf={confidence:.2f})"
        else:
            reason = f"VEB不开火(Q={q_hold:.2f}, conf={1-confidence:.2f})"

        return should_fire, confidence, reason

    def batch_should_fire(self, pairs: List[Tuple[Dict, Dict]],
                          use_elite: bool = True) -> List[Tuple[bool, float, str]]:
        """批量判断是否应该开火"""
        if not pairs:
            return []

        states = np.array([self.extract_state(s, t) for s, t in pairs], dtype=np.float32)
        states_tensor = torch.FloatTensor(states).to(self.device)

        if use_elite and len(self.fitness_history) > 0:
            fitness_scores = self.evaluate_population()
            best_idx = np.argmax(fitness_scores)
            q_net = self.population[best_idx][0]
        else:
            q_net = self.rl_q_net

        with torch.no_grad():
            q_values = q_net(states_tensor)
            actions = q_values.argmax(dim=1)
            confidences = torch.softmax(q_values, dim=1)[:, 1]

        results = []
        for i, (action, conf) in enumerate(zip(actions, confidences)):
            should_fire = action.item() == 1
            confidence = conf.item()
            if should_fire:
                reason = f"VEB开火(conf={confidence:.2f})"
            else:
                reason = f"VEB不开火(conf={1-confidence:.2f})"
            results.append((should_fire, confidence, reason))

        return results

    # ==================== 协同攻击决策 ====================

    def coordinated_attack_decision(self, shooters: List[Dict], target: Dict,
                                     min_shooters: int = 2) -> List[Tuple[Dict, bool, str]]:
        """
        协同攻击决策 - 多架飞机同时攻击同一目标

        核心逻辑:
        1. 找出所有能攻击该目标的射手 (在射程内且有弹药)
        2. 如果满足最小射手数量，则全部同时开火
        3. 否则不开火，等待更好的协同时机
        4. 特殊情况: 如果只剩最后一架有弹药的飞机，不发射!
           - 单发攻击几乎必定被躲避，浪费弹药
           - 保留弹药去执行占点等其他策略更有价值

        Args:
            shooters: 所有己方飞机列表
            target: 目标信息
            min_shooters: 最少需要多少架飞机协同 (默认2)

        Returns:
            List[(shooter, should_fire, reason)] 每架飞机的开火决策
        """
        t_lon = target.get('longitude', target.get('X', 0))
        t_lat = target.get('latitude', target.get('Y', 0))
        t_alt = target.get('altitude', target.get('Alt', 5000))
        t_speed = target.get('speed', 300)
        t_heading = target.get('heading', 0)

        # ========== 第零步: 检查是否只剩最后一架有弹药的飞机 ==========
        # 如果全队只剩一架有弹药，不发射! 保留弹药去占点
        shooters_with_ammo = []
        for shooter in shooters:
            weapons = shooter.get('weapons', [])
            if weapons and weapons[0].get('quantity', 0) > 0:
                shooters_with_ammo.append(shooter)

        if len(shooters_with_ammo) <= 1:
            # 只剩最后一架有弹药的飞机，全部不开火
            results = []
            for shooter in shooters:
                weapons = shooter.get('weapons', [])
                if weapons and weapons[0].get('quantity', 0) > 0:
                    reason = "最后一架有弹药飞机，保留弹药执行占点策略，不发射"
                else:
                    reason = "无弹药"
                results.append((shooter, False, reason))
            return results

        # ========== 第一步: 筛选有效射手 ==========
        valid_shooters = []  # (shooter, distance, in_nez, aspect_angle)

        for shooter in shooters:
            # 检查弹药
            weapons = shooter.get('weapons', [])
            if not weapons or weapons[0].get('quantity', 0) <= 0:
                continue

            # 计算距离
            s_lon = shooter.get('longitude', 0)
            s_lat = shooter.get('latitude', 0)
            s_alt = shooter.get('altitude', 5000)
            s_speed = shooter.get('speed', 300)
            is_manned = shooter.get('type') == '有人机'

            h_dist = self._haversine_distance(s_lon, s_lat, t_lon, t_lat)
            dist_3d = math.sqrt(h_dist**2 + (s_alt - t_alt)**2)

            # 计算姿态角
            aspect_angle = self._calculate_aspect_angle(
                s_lon, s_lat, shooter.get('heading', 0),
                t_lon, t_lat, t_heading
            )

            # 计算NEZ
            nez, max_range = self._calculate_nez(
                dist_3d, aspect_angle, s_alt, t_alt,
                s_speed, t_speed, is_manned
            )

            # 在射程内
            if dist_3d <= max_range:
                in_nez = dist_3d <= nez
                valid_shooters.append({
                    'shooter': shooter,
                    'distance': dist_3d,
                    'in_nez': in_nez,
                    'aspect_angle': aspect_angle,
                    'nez': nez,
                    'max_range': max_range
                })

        # ========== 第二步: 协同攻击决策 ==========
        results = []

        # 统计在NEZ内的射手
        shooters_in_nez = [s for s in valid_shooters if s['in_nez']]

        if len(shooters_in_nez) >= min_shooters:
            # 满足协同条件，所有在NEZ内的射手同时开火
            for s_info in valid_shooters:
                shooter = s_info['shooter']
                if s_info['in_nez']:
                    reason = f"协同攻击! {len(shooters_in_nez)}机齐射, 距离{s_info['distance']:.0f}m, NEZ内"
                    results.append((shooter, True, reason))
                else:
                    reason = f"不在NEZ内(距离{s_info['distance']:.0f}m > NEZ{s_info['nez']:.0f}m), 等待"
                    results.append((shooter, False, reason))

            # 没有有效射手的飞机
            valid_shooter_ids = {s['shooter'].get('id') for s in valid_shooters}
            for shooter in shooters:
                if shooter.get('id') not in valid_shooter_ids:
                    weapons = shooter.get('weapons', [])
                    if not weapons or weapons[0].get('quantity', 0) <= 0:
                        results.append((shooter, False, "无弹药"))
                    else:
                        results.append((shooter, False, "超出射程"))

        else:
            # 不满足协同条件，全部等待
            reason = f"等待协同(当前{len(shooters_in_nez)}机在NEZ, 需要{min_shooters}机)"
            for shooter in shooters:
                results.append((shooter, False, reason))

        return results

    def get_best_coordinated_targets(self, shooters: List[Dict], targets: List[Dict],
                                      min_shooters: int = 2) -> List[Dict]:
        """
        找出最适合协同攻击的目标

        优先级:
        1. 有人机目标 (高价值)
        2. 有更多射手能同时攻击的目标
        3. 已经有导弹在攻击的目标 (继续压制)

        特殊情况:
        - 如果只剩最后一架有弹药的飞机，返回空列表 (不攻击，去占点)

        Args:
            shooters: 所有己方飞机
            targets: 所有敌方目标
            min_shooters: 最少协同射手数

        Returns:
            按优先级排序的目标列表，每个目标附带可协同攻击的射手信息
            如果只剩最后一架有弹药飞机，返回包含 'last_shooter_strategy' 标记的特殊结果
        """
        # ========== 检查是否只剩最后一架有弹药的飞机 ==========
        shooters_with_ammo = [
            s for s in shooters
            if s.get('weapons', []) and s.get('weapons', [])[0].get('quantity', 0) > 0
        ]

        if len(shooters_with_ammo) <= 1:
            # 只剩最后一架有弹药的飞机，不攻击，建议去占点
            return [{
                'target': None,
                'score': 0,
                'shooters_in_nez': [],
                'shooters_in_range': [],
                'can_coordinate': False,
                'last_shooter_strategy': True,  # 特殊标记: 最后一架飞机策略
                'recommendation': '只剩最后一架有弹药飞机，建议执行占点策略'
            }]

        target_scores = []

        for target in targets:
            t_lon = target.get('longitude', target.get('X', 0))
            t_lat = target.get('latitude', target.get('Y', 0))
            t_alt = target.get('altitude', target.get('Alt', 5000))
            t_speed = target.get('speed', 300)
            t_heading = target.get('heading', 0)
            is_manned_target = target.get('platform_entity_type') == '有人机'
            is_fired_num = target.get('is_fired_num', 0)

            # 统计能攻击该目标的射手
            shooters_in_nez = []
            shooters_in_range = []

            for shooter in shooters:
                weapons = shooter.get('weapons', [])
                if not weapons or weapons[0].get('quantity', 0) <= 0:
                    continue

                s_lon = shooter.get('longitude', 0)
                s_lat = shooter.get('latitude', 0)
                s_alt = shooter.get('altitude', 5000)
                s_speed = shooter.get('speed', 300)
                is_manned = shooter.get('type') == '有人机'

                h_dist = self._haversine_distance(s_lon, s_lat, t_lon, t_lat)
                dist_3d = math.sqrt(h_dist**2 + (s_alt - t_alt)**2)

                aspect_angle = self._calculate_aspect_angle(
                    s_lon, s_lat, shooter.get('heading', 0),
                    t_lon, t_lat, t_heading
                )

                nez, max_range = self._calculate_nez(
                    dist_3d, aspect_angle, s_alt, t_alt,
                    s_speed, t_speed, is_manned
                )

                if dist_3d <= nez:
                    shooters_in_nez.append(shooter)
                elif dist_3d <= max_range:
                    shooters_in_range.append(shooter)

            # 计算目标优先级得分
            score = 0
            score += len(shooters_in_nez) * 100      # 每个NEZ内射手+100
            score += len(shooters_in_range) * 20    # 每个射程内射手+20
            score += 200 if is_manned_target else 0  # 有人机+200
            score += is_fired_num * 50               # 已被攻击+50 (继续压制)

            if len(shooters_in_nez) >= min_shooters:
                score += 500  # 满足协同条件大加分

            target_scores.append({
                'target': target,
                'score': score,
                'shooters_in_nez': shooters_in_nez,
                'shooters_in_range': shooters_in_range,
                'can_coordinate': len(shooters_in_nez) >= min_shooters
            })

        # 按得分排序
        target_scores.sort(key=lambda x: x['score'], reverse=True)

        return target_scores

    # ==================== 存储经验 ====================

    def store_transition(self, shooter: Dict, target: Dict, action: int,
                         reward: float, next_shooter: Dict, next_target: Dict, done: bool):
        """存储经验"""
        state = self.extract_state(shooter, target)
        next_state = self.extract_state(next_shooter, next_target)
        self.replay_buffer.add(state, action, reward, next_state, done)

    # ==================== 保存/加载 ====================

    def save(self, filepath: str):
        """保存模型"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # 保存所有种群个体
        population_state = []
        for q_net, target_net in self.population:
            population_state.append({
                'q_net': q_net.state_dict(),
                'target_net': target_net.state_dict()
            })

        checkpoint = {
            'population': population_state,
            'rl_q_net': self.rl_q_net.state_dict(),
            'rl_target_net': self.rl_target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_steps': self.train_steps,
            'generation': self.generation,
            'config': self.config,
            'fitness_history': self.fitness_history
        }

        if self.config.ea_type == 'CEM':
            checkpoint['cem_mean'] = self.cem_mean
            checkpoint['cem_std'] = self.cem_std

        torch.save(checkpoint, filepath)
        print(f"[VEB-RL] 模型已保存: {filepath}")

    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        # 加载种群
        for i, state in enumerate(checkpoint['population']):
            if i < len(self.population):
                self.population[i][0].load_state_dict(state['q_net'])
                self.population[i][1].load_state_dict(state['target_net'])

        # 加载RL个体
        self.rl_q_net.load_state_dict(checkpoint['rl_q_net'])
        self.rl_target_net.load_state_dict(checkpoint['rl_target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.train_steps = checkpoint['train_steps']
        self.generation = checkpoint['generation']
        self.fitness_history = checkpoint.get('fitness_history', [])

        if self.config.ea_type == 'CEM' and 'cem_mean' in checkpoint:
            self.cem_mean = checkpoint['cem_mean']
            self.cem_std = checkpoint['cem_std']

        print(f"[VEB-RL] 模型已加载: {filepath}, 代数: {self.generation}, 步数: {self.train_steps}")


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("VEB-RL火控系统测试")
    print("=" * 60)

    # 创建VEB火控
    config = VEBConfig(
        population_size=5,
        elite_num=2,
        ea_type='GA',
        device='cpu'
    )
    veb = VEBFireControl(config)

    # 测试状态提取
    shooter = {
        'longitude': 146.1,
        'latitude': 33.3,
        'speed': 350,
        'heading': 0,
        'altitude': 5000,
        'type': '无人机'
    }

    target = {
        'longitude': 146.2,
        'latitude': 33.4,
        'speed': 300,
        'heading': 180,
        'altitude': 4500,
        'platform_entity_type': '有人机'
    }

    # 提取状态
    state = veb.extract_state(shooter, target)
    print(f"\n状态向量 (9维): {state}")

    # 添加一些模拟数据
    for _ in range(1000):
        s = np.random.randn(9).astype(np.float32)
        a = np.random.randint(2)
        r = np.random.randn()
        ns = np.random.randn(9).astype(np.float32)
        d = False
        veb.replay_buffer.add(s, a, r, ns, d)

    # 测试适应度评估
    fitness_scores = veb.evaluate_population()
    print(f"\n种群适应度: {fitness_scores}")

    # 测试进化
    veb.evolve_population()
    print(f"\n进化后适应度: {veb.evaluate_population()}")

    # 测试RL更新
    stats = veb.update_rl()
    print(f"\nRL更新统计: {stats}")

    # 测试推理
    should_fire, confidence, reason = veb.should_fire(shooter, target)
    print(f"\n开火决策:")
    print(f"  应该开火: {should_fire}")
    print(f"  置信度: {confidence:.3f}")
    print(f"  原因: {reason}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
