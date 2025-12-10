"""
VEB-RL导弹躲避策略核心模块

基于VEB-RL论文实现的导弹规避决策系统
结合进化算法和强化学习来学习最优规避机动

动作空间设计:
- 离散化的机动指令: 方向 × 高度变化 × 速度
- 或者直接输出机动点偏移量

奖励设计:
- 成功躲避: +1.0
- 被击中: -1.0
- 每步存活: +0.01
- 与导弹保持距离: 距离相关奖励
"""

import math
import os
import copy
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ==================== 配置类 ====================

@dataclass
class VEBEvasionConfig:
    """VEB-RL躲避策略配置"""
    # 状态空间 (更新: 12维 -> 14维，增加速度正交分解和更多物理信息)
    state_dim: int = 14

    # 动作空间 (离散化)
    # 方向: 8个 (N, NE, E, SE, S, SW, W, NW)
    # 高度: 3个 (上升, 保持, 下降)
    # 速度: 3个 (加速, 保持, 减速)
    # 总共: 8 × 3 × 3 = 72个离散动作
    # 简化版: 9个机动方向 (8方向 + 保持)
    action_dim: int = 9

    # 网络配置
    hidden_dims: Tuple[int, ...] = (256, 256, 128)
    dueling: bool = True

    # 种群配置
    population_size: int = 10
    elite_num: int = 2

    # 进化配置
    ea_type: str = 'GA'
    mutation_prob: float = 0.3
    mutation_strength: float = 0.1
    crossover_prob: float = 0.5
    tournament_size: int = 3

    # RL配置
    lr: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 128
    buffer_capacity: int = 200000
    target_update_freq: int = 500
    population_target_update_freq: int = 20

    # 适应度评估
    fitness_sample_size: int = 2048

    # 躲避参数
    evasion_distance_threshold: float = 3000  # 需要躲避的距离阈值
    safe_distance: float = 5000               # 安全距离

    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ==================== Q网络 ====================

class EvasionQNetwork(nn.Module):
    """
    躲避策略Q网络

    使用Dueling架构分离状态价值和动作优势
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, ...]):
        super().__init__()

        # 特征提取
        layers = []
        in_dim = state_dim
        for hidden_dim in hidden_dims[:-1]:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        self.features = nn.Sequential(*layers)

        last_hidden = hidden_dims[-1]

        # 价值流
        self.value_hidden = nn.Linear(in_dim, last_hidden)
        self.value = nn.Linear(last_hidden, 1)

        # 优势流
        self.advantage_hidden = nn.Linear(in_dim, last_hidden)
        self.advantage = nn.Linear(last_hidden, action_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.features(state)

        # 价值流
        value = F.relu(self.value_hidden(features))
        value = self.value(value)

        # 优势流
        advantage = F.relu(self.advantage_hidden(features))
        advantage = self.advantage(advantage)

        # 组合
        q_value = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_value


# ==================== 经验回放 ====================

class EvasionReplayBuffer:
    """躲避经验回放缓冲区"""

    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: str) -> Dict[str, torch.Tensor]:
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


# ==================== VEB躲避系统 ====================

class VEBEvasion:
    """
    VEB-RL导弹躲避系统

    学习最优的导弹规避机动策略
    """

    def __init__(self, config: VEBEvasionConfig = None):
        self.config = config or VEBEvasionConfig()
        self.device = torch.device(self.config.device)

        # ==================== 初始化种群 ====================
        self.population = []
        for _ in range(self.config.population_size):
            q_net = EvasionQNetwork(
                self.config.state_dim,
                self.config.action_dim,
                self.config.hidden_dims
            ).to(self.device)

            target_net = EvasionQNetwork(
                self.config.state_dim,
                self.config.action_dim,
                self.config.hidden_dims
            ).to(self.device)

            target_net.load_state_dict(q_net.state_dict())
            self.population.append((q_net, target_net))

        # ==================== RL个体 ====================
        self.rl_q_net = EvasionQNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dims
        ).to(self.device)

        self.rl_target_net = EvasionQNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dims
        ).to(self.device)

        self.rl_target_net.load_state_dict(self.rl_q_net.state_dict())
        self.optimizer = optim.Adam(self.rl_q_net.parameters(), lr=self.config.lr)

        # ==================== 经验回放 ====================
        self.replay_buffer = EvasionReplayBuffer(
            self.config.buffer_capacity,
            self.config.state_dim
        )

        # ==================== 训练统计 ====================
        self.train_steps = 0
        self.generation = 0
        self.fitness_history = []

        # ==================== 动作映射 ====================
        # 9个方向: 8个方向 + 保持
        # 0-7: 8个方向 (N, NE, E, SE, S, SW, W, NW)
        # 8: 保持当前航向
        self.direction_offsets = [
            (0, 1),     # N
            (1, 1),     # NE
            (1, 0),     # E
            (1, -1),    # SE
            (0, -1),    # S
            (-1, -1),   # SW
            (-1, 0),    # W
            (-1, 1),    # NW
            (0, 0)      # 保持
        ]

    # ==================== 状态提取 ====================

    def extract_state(self, unit: Dict, missiles: List[Dict]) -> np.ndarray:
        """
        提取躲避状态向量

        基于真实态势数据结构设计，充分利用速度分量信息

        状态空间 (14维):
        0. 己方横向逃逸速度 - 垂直于导弹来袭方向的速度分量 (核心!)
        1. 己方径向速度 - 远离导弹的速度分量 (正=远离, 负=靠近)
        2. 导弹相对方位sin - 导弹在己方的哪个方向 (相对于己方航向)
        3. 导弹相对方位cos - cos>0表示导弹在前方
        4. 导弹3D距离 (归一化)
        5. 导弹接近速度 (利用速度分量精确计算)
        6. 导弹速度 (归一化)
        7. 预计碰撞时间TTC (归一化)
        8. 导弹数量 (归一化)
        9. 威胁等级 (综合距离和接近速度)
        10. 高度差 (己方 - 导弹, 正=己方更高)
        11. 己方机动强度 (根据roll角判断)
        12. 导弹剩余能量估计
        13. 己方是否有高度机动空间 (太低不能俯冲)

        物理意义:
        - 横向逃逸速度(V_lateral)是躲避的核心 - 越大越容易躲开
        - 径向速度(V_radial)决定能否拉开距离
        - 利用velocity_x/y/z直接计算，比用heading更准确
        """
        # ========== 己方信息 ==========
        u_lon = unit.get('longitude', 146.0)
        u_lat = unit.get('latitude', 33.3)
        u_alt = unit.get('altitude', 5000)
        u_speed = unit.get('speed', 300)
        # 速度分量 (东为正x, 北为正y, 上为正z)
        u_vx = unit.get('velocity_x', 0)
        u_vy = unit.get('velocity_y', 0)
        u_vz = unit.get('velocity_z', 0)
        u_heading = unit.get('heading', 0)  # 弧度
        u_roll = unit.get('roll', 0)  # 横滚角

        # 航向转弧度 (数据已经是弧度)
        if isinstance(u_heading, (int, float)):
            u_heading_rad = u_heading if abs(u_heading) <= math.pi else math.radians(u_heading)
        else:
            u_heading_rad = u_heading

        # ========== 无导弹威胁 ==========
        if not missiles:
            return np.array([
                0,                        # 0: 横向速度 (无威胁方向)
                u_speed / 500,            # 1: 当前速度作为径向速度
                0,                        # 2: 方位sin
                1,                        # 3: 方位cos (假设前方)
                1.0,                      # 4: 距离远
                0,                        # 5: 无接近
                0,                        # 6: 无导弹速度
                1.0,                      # 7: TTC长
                0,                        # 8: 无导弹
                0,                        # 9: 无威胁
                0,                        # 10: 无高度差
                0,                        # 11: 未机动
                0,                        # 12: 无能量威胁
                1.0 if u_alt > 2000 else u_alt / 2000  # 13: 高度空间
            ], dtype=np.float32)

        # ========== 找最近的导弹 ==========
        min_dist = float('inf')
        nearest_missile = None

        for missile in missiles:
            m_lon = missile.get('longitude', 0)
            m_lat = missile.get('latitude', 0)
            m_alt = missile.get('altitude', u_alt)
            # 3D距离
            h_dist = self._haversine_distance(u_lon, u_lat, m_lon, m_lat)
            v_dist = abs(u_alt - m_alt)
            dist_3d = math.sqrt(h_dist**2 + v_dist**2)

            if dist_3d < min_dist:
                min_dist = dist_3d
                nearest_missile = missile

        # ========== 最近导弹信息 ==========
        m_lon = nearest_missile.get('longitude', 0)
        m_lat = nearest_missile.get('latitude', 0)
        m_alt = nearest_missile.get('altitude', u_alt)
        m_speed = nearest_missile.get('speed', 1200)
        # 导弹速度分量 (track数据用v_x/v_y/v_z)
        m_vx = nearest_missile.get('v_x', nearest_missile.get('velocity_x', 0))
        m_vy = nearest_missile.get('v_y', nearest_missile.get('velocity_y', 0))
        m_vz = nearest_missile.get('v_z', nearest_missile.get('velocity_z', 0))

        # ========== 核心: 己方速度正交分解 ==========
        v_lateral, v_radial = self._decompose_unit_velocity_3d(
            u_lon, u_lat, u_alt, u_vx, u_vy, u_vz,
            m_lon, m_lat, m_alt
        )

        # ========== 导弹相对方位角 ==========
        dx = m_lon - u_lon
        dy = m_lat - u_lat
        bearing_to_missile = math.atan2(dx, dy)  # 弧度，北为0

        # 相对于己方航向的方位
        relative_bearing = bearing_to_missile - u_heading_rad
        # 归一化到 [-π, π]
        while relative_bearing > math.pi:
            relative_bearing -= 2 * math.pi
        while relative_bearing < -math.pi:
            relative_bearing += 2 * math.pi

        bearing_sin = math.sin(relative_bearing)
        bearing_cos = math.cos(relative_bearing)

        # ========== 利用速度分量精确计算接近速度 ==========
        dx_m = (m_lon - u_lon) * 111000 * math.cos(math.radians((u_lat + m_lat) / 2))
        dy_m = (m_lat - u_lat) * 111000
        dz_m = m_alt - u_alt
        dist_m = math.sqrt(dx_m**2 + dy_m**2 + dz_m**2)

        if dist_m > 0:
            ux, uy, uz = dx_m / dist_m, dy_m / dist_m, dz_m / dist_m
            # 相对速度 (导弹速度 - 己方速度)
            rel_vx = m_vx - u_vx
            rel_vy = m_vy - u_vy
            rel_vz = m_vz - u_vz
            # 接近速度 (正值表示导弹在接近)
            closure_rate = rel_vx * ux + rel_vy * uy + rel_vz * uz
        else:
            closure_rate = m_speed  # 默认导弹速度

        # ========== 预计碰撞时间 TTC ==========
        if closure_rate > 10:
            time_to_collision = min_dist / closure_rate
        else:
            time_to_collision = 999

        # ========== 威胁等级 ==========
        threat_level = max(0, 1 - min_dist / self.config.safe_distance)
        if closure_rate > 800:
            threat_level = min(1.0, threat_level * 1.5)

        # ========== 高度差 ==========
        alt_diff = u_alt - m_alt

        # ========== 己方机动强度 ==========
        maneuver_intensity = min(abs(u_roll) / 1.4, 1.0)

        # ========== 导弹剩余能量估计 ==========
        # 假设导弹初始射程30km，根据当前速度估算剩余能量
        energy_remaining = min(m_speed / 1200, 1.0) * 0.8 + 0.2

        # ========== 高度机动空间 ==========
        # 太低不能俯冲，太高不能爬升
        altitude_margin = min(u_alt / 2000, 1.0)  # 2000米以下受限

        # ========== 构建状态向量 ==========
        state = np.array([
            np.clip(v_lateral / 400, -1, 1),              # 0: 横向逃逸速度 (核心!)
            np.clip(v_radial / 400, -1, 1),               # 1: 径向速度 (正=远离)
            bearing_sin,                                   # 2: 导弹方位sin
            bearing_cos,                                   # 3: 导弹方位cos
            min(min_dist / 10000, 1.0),                   # 4: 导弹距离
            np.clip(closure_rate / 1500, -1, 1),          # 5: 接近速度
            min(m_speed / 1500, 1.0),                     # 6: 导弹速度
            min(time_to_collision / 30, 1.0),             # 7: TTC
            min(len(missiles) / 5, 1.0),                  # 8: 导弹数量
            threat_level,                                  # 9: 威胁等级
            np.clip(alt_diff / 3000, -1, 1),              # 10: 高度差
            maneuver_intensity,                            # 11: 机动强度
            energy_remaining,                              # 12: 导弹能量
            altitude_margin                                # 13: 高度空间
        ], dtype=np.float32)

        return state

    def _decompose_unit_velocity_3d(self, u_lon: float, u_lat: float, u_alt: float,
                                     u_vx: float, u_vy: float, u_vz: float,
                                     m_lon: float, m_lat: float, m_alt: float) -> Tuple[float, float]:
        """
        利用3D速度分量直接计算己方相对于导弹来袭方向的横向和径向速度

        这是躲避决策的核心!

        Args:
            u_lon, u_lat, u_alt: 己方位置
            u_vx, u_vy, u_vz: 己方速度分量 (东/北/上)
            m_lon, m_lat, m_alt: 导弹位置

        Returns:
            (v_lateral, v_radial): 横向速度(躲避能力)和径向速度(正=远离导弹)

        最佳躲避策略:
        - 最大化 V_lateral (垂直于导弹来袭方向飞行)
        - 保持正的 V_radial (远离导弹)
        """
        # 计算导弹到己方的方向向量 (这是导弹来袭方向)
        dx_m = (u_lon - m_lon) * 111000 * math.cos(math.radians((u_lat + m_lat) / 2))
        dy_m = (u_lat - m_lat) * 111000
        dz_m = u_alt - m_alt

        dist = math.sqrt(dx_m**2 + dy_m**2 + dz_m**2)

        if dist < 1:
            return 0, 0

        # 导弹来袭方向的单位向量 (从导弹指向己方)
        ux, uy, uz = dx_m / dist, dy_m / dist, dz_m / dist

        # 己方速度在来袭方向的投影 (正=远离导弹)
        v_radial = u_vx * ux + u_vy * uy + u_vz * uz

        # 己方速度的横向分量
        u_speed = math.sqrt(u_vx**2 + u_vy**2 + u_vz**2)
        v_lateral_sq = u_speed**2 - v_radial**2
        v_lateral = math.sqrt(max(0, v_lateral_sq))

        return v_lateral, v_radial

    def _haversine_distance(self, lon1, lat1, lon2, lat2) -> float:
        """计算两点间距离"""
        R = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def _calculate_closure_rate(self, u_lon, u_lat, u_speed, u_heading,
                                 m_lon, m_lat, m_speed, m_heading) -> float:
        """计算接近率"""
        dx = m_lon - u_lon
        dy = m_lat - u_lat
        bearing_to_missile = math.degrees(math.atan2(dx, dy)) % 360

        u_hdg = math.degrees(u_heading) % 360 if isinstance(u_heading, float) else u_heading % 360
        m_hdg = math.degrees(m_heading) % 360 if isinstance(m_heading, float) else m_heading % 360

        # 导弹接近速度
        m_angle = math.radians(bearing_to_missile + 180 - m_hdg)
        missile_closure = m_speed * math.cos(m_angle)

        # 己方远离速度
        u_angle = math.radians(bearing_to_missile - u_hdg)
        unit_closure = u_speed * math.cos(u_angle)

        return missile_closure - unit_closure

    # ==================== 动作转换 ====================

    def action_to_maneuver(self, action: int, unit: Dict) -> Tuple[float, float, float]:
        """
        将动作转换为机动指令

        Args:
            action: 动作索引 (0-8)
            unit: 当前单位信息

        Returns:
            (target_lat, target_lon, target_alt) 目标点
        """
        u_lon = unit.get('longitude', 146.0)
        u_lat = unit.get('latitude', 33.3)
        u_alt = unit.get('altitude', 5000)
        u_heading = unit.get('heading', 0)

        # 获取方向偏移
        dx, dy = self.direction_offsets[action]

        if action == 8:
            # 保持当前方向，继续前进
            heading_rad = math.radians(u_heading) if isinstance(u_heading, (int, float)) else u_heading
            dx = math.sin(heading_rad)
            dy = math.cos(heading_rad)

        # 计算目标点 (机动距离约2km)
        maneuver_distance = 2000  # 米
        lat_offset = (dy * maneuver_distance) / 111000  # 纬度偏移
        lon_offset = (dx * maneuver_distance) / (111000 * math.cos(math.radians(u_lat)))

        target_lat = u_lat + lat_offset
        target_lon = u_lon + lon_offset
        target_alt = u_alt  # 暂时保持高度

        return target_lat, target_lon, target_alt

    # ==================== 适应度评估 ====================

    def calculate_fitness(self, q_net: nn.Module, target_net: nn.Module) -> float:
        """计算适应度: 负TD误差"""
        if len(self.replay_buffer) < self.config.batch_size:
            return float('-inf')

        batch = self.replay_buffer.sample(self.config.fitness_sample_size, self.device)

        with torch.no_grad():
            q_values = q_net(batch['states'])
            current_q = q_values.gather(1, batch['actions'])

            next_q = target_net(batch['next_states']).max(dim=1, keepdim=True)[0]
            target_q = batch['rewards'] + self.config.gamma * (1 - batch['dones']) * next_q

            td_error = (target_q - current_q).pow(2).mean()

        return -td_error.item()

    # ==================== 进化 ====================

    def evolve_population(self):
        """进化种群"""
        self.generation += 1

        # 评估适应度
        fitness_scores = []
        for q_net, target_net in self.population:
            fitness = self.calculate_fitness(q_net, target_net)
            fitness_scores.append(fitness)

        # RL个体适应度
        rl_fitness = self.calculate_fitness(self.rl_q_net, self.rl_target_net)

        # 注入RL个体
        min_idx = np.argmin(fitness_scores)
        if rl_fitness > fitness_scores[min_idx]:
            self.population[min_idx][0].load_state_dict(self.rl_q_net.state_dict())
            self.population[min_idx][1].load_state_dict(self.rl_target_net.state_dict())
            fitness_scores[min_idx] = rl_fitness

        # GA进化
        sorted_indices = np.argsort(fitness_scores)[::-1]
        self._evolve_ga(sorted_indices, fitness_scores)

        # 更新目标网络
        if self.generation % self.config.population_target_update_freq == 0:
            for q_net, target_net in self.population:
                target_net.load_state_dict(q_net.state_dict())

        self.fitness_history.append({
            'generation': self.generation,
            'mean_fitness': np.mean(fitness_scores),
            'max_fitness': np.max(fitness_scores),
            'rl_fitness': rl_fitness
        })

    def _evolve_ga(self, sorted_indices: np.ndarray, fitness_scores: List[float]):
        """GA进化操作"""
        elite_indices = sorted_indices[:self.config.elite_num]
        non_elite_indices = sorted_indices[self.config.elite_num:]

        # 锦标赛选择
        winners = []
        for _ in range(len(non_elite_indices) // 2):
            candidates = random.sample(list(sorted_indices), self.config.tournament_size)
            winner = max(candidates, key=lambda x: fitness_scores[x])
            winners.append(winner)

        # 交叉和变异
        for idx in non_elite_indices:
            parent1_idx = random.choice(list(elite_indices) + winners)
            parent2_idx = random.choice(list(elite_indices) + winners)

            # 交叉
            if random.random() < self.config.crossover_prob:
                self._crossover_networks(
                    self.population[parent1_idx][0],
                    self.population[parent2_idx][0],
                    self.population[idx][0]
                )
            else:
                self.population[idx][0].load_state_dict(
                    self.population[parent1_idx][0].state_dict()
                )

            # 变异
            if random.random() < self.config.mutation_prob:
                self._mutate_network(self.population[idx][0])

    def _crossover_networks(self, parent1: nn.Module, parent2: nn.Module, child: nn.Module):
        """网络交叉"""
        for (n1, p1), (n2, p2), (nc, pc) in zip(
            parent1.named_parameters(),
            parent2.named_parameters(),
            child.named_parameters()
        ):
            mask = torch.rand_like(p1) > 0.5
            pc.data.copy_(torch.where(mask, p1.data, p2.data))

    def _mutate_network(self, net: nn.Module):
        """网络变异"""
        with torch.no_grad():
            for param in net.parameters():
                noise = torch.randn_like(param) * self.config.mutation_strength
                param.add_(noise)

    # ==================== RL更新 ====================

    def update_rl(self) -> Dict[str, float]:
        """DQN更新"""
        if len(self.replay_buffer) < self.config.batch_size:
            return {}

        self.train_steps += 1
        batch = self.replay_buffer.sample(self.config.batch_size, self.device)

        # Double DQN
        q_values = self.rl_q_net(batch['states'])
        current_q = q_values.gather(1, batch['actions'])

        with torch.no_grad():
            next_actions = self.rl_q_net(batch['next_states']).argmax(dim=1, keepdim=True)
            next_q = self.rl_target_net(batch['next_states']).gather(1, next_actions)
            target_q = batch['rewards'] + self.config.gamma * (1 - batch['dones']) * next_q

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rl_q_net.parameters(), 10.0)
        self.optimizer.step()

        # 更新目标网络
        if self.train_steps % self.config.target_update_freq == 0:
            self.rl_target_net.load_state_dict(self.rl_q_net.state_dict())

        return {'loss': loss.item(), 'q_value': current_q.mean().item()}

    # ==================== 推理接口 ====================

    def get_evasion_action(self, unit: Dict, missiles: List[Dict],
                           epsilon: float = 0.0) -> Tuple[int, Tuple[float, float, float]]:
        """
        获取躲避动作

        Args:
            unit: 己方单位信息
            missiles: 来袭导弹列表
            epsilon: 探索率

        Returns:
            (action, maneuver_point) 动作和机动目标点
        """
        state = self.extract_state(unit, missiles)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # epsilon-greedy
        if random.random() < epsilon:
            action = random.randint(0, self.config.action_dim - 1)
        else:
            # 使用最佳个体
            if self.fitness_history:
                fitness_scores = [self.calculate_fitness(q, t) for q, t in self.population]
                best_idx = np.argmax(fitness_scores)
                q_net = self.population[best_idx][0]
            else:
                q_net = self.rl_q_net

            with torch.no_grad():
                q_values = q_net(state_tensor)
                action = q_values.argmax(dim=1).item()

        maneuver_point = self.action_to_maneuver(action, unit)
        return action, maneuver_point

    def should_evade(self, unit: Dict, missiles: List[Dict]) -> bool:
        """判断是否需要躲避"""
        if not missiles:
            return False

        u_lon = unit.get('longitude', 0)
        u_lat = unit.get('latitude', 0)

        for missile in missiles:
            m_lon = missile.get('longitude', 0)
            m_lat = missile.get('latitude', 0)
            dist = self._haversine_distance(u_lon, u_lat, m_lon, m_lat)

            if dist < self.config.evasion_distance_threshold:
                return True

        return False

    # ==================== 存储经验 ====================

    def store_transition(self, unit: Dict, missiles: List[Dict], action: int,
                         reward: float, next_unit: Dict, next_missiles: List[Dict], done: bool):
        """存储躲避经验"""
        state = self.extract_state(unit, missiles)
        next_state = self.extract_state(next_unit, next_missiles)
        self.replay_buffer.add(state, action, reward, next_state, done)

    # ==================== 奖励计算 ====================

    @staticmethod
    def calculate_reward(evaded: bool, hit: bool, distance_to_missile: float,
                         safe_distance: float = 5000) -> float:
        """
        计算躲避奖励

        Args:
            evaded: 是否成功躲避
            hit: 是否被击中
            distance_to_missile: 与最近导弹距离
            safe_distance: 安全距离

        Returns:
            奖励值
        """
        if hit:
            return -1.0

        if evaded:
            return 1.0

        # 距离奖励: 越远越好
        distance_reward = min(distance_to_missile / safe_distance, 1.0) * 0.1

        # 存活奖励
        survival_reward = 0.01

        return distance_reward + survival_reward

    # ==================== 保存/加载 ====================

    def save(self, filepath: str):
        """保存模型"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

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

        torch.save(checkpoint, filepath)
        print(f"[VEB-Evasion] 模型已保存: {filepath}")

    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        for i, state in enumerate(checkpoint['population']):
            if i < len(self.population):
                self.population[i][0].load_state_dict(state['q_net'])
                self.population[i][1].load_state_dict(state['target_net'])

        self.rl_q_net.load_state_dict(checkpoint['rl_q_net'])
        self.rl_target_net.load_state_dict(checkpoint['rl_target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.train_steps = checkpoint['train_steps']
        self.generation = checkpoint['generation']
        self.fitness_history = checkpoint.get('fitness_history', [])

        print(f"[VEB-Evasion] 模型已加载: {filepath}")


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("VEB-RL导弹躲避系统测试")
    print("=" * 60)

    config = VEBEvasionConfig(
        population_size=5,
        elite_num=2,
        device='cpu'
    )
    veb_evasion = VEBEvasion(config)

    # 测试单位
    unit = {
        'longitude': 146.1,
        'latitude': 33.3,
        'altitude': 5000,
        'speed': 350,
        'heading': 45
    }

    # 测试导弹
    missiles = [
        {
            'longitude': 146.12,
            'latitude': 33.32,
            'altitude': 5000,
            'speed': 1200,
            'heading': 225
        }
    ]

    # 提取状态
    state = veb_evasion.extract_state(unit, missiles)
    print(f"\n状态向量 (12维): {state}")

    # 判断是否需要躲避
    should_evade = veb_evasion.should_evade(unit, missiles)
    print(f"\n需要躲避: {should_evade}")

    # 获取躲避动作
    action, maneuver_point = veb_evasion.get_evasion_action(unit, missiles, epsilon=0.1)
    print(f"\n躲避动作: {action}")
    print(f"机动目标点: {maneuver_point}")

    # 添加模拟数据
    for _ in range(1000):
        s = np.random.randn(12).astype(np.float32)
        a = np.random.randint(9)
        r = np.random.randn()
        ns = np.random.randn(12).astype(np.float32)
        d = False
        veb_evasion.replay_buffer.add(s, a, r, ns, d)

    # 测试进化
    veb_evasion.evolve_population()
    print(f"\n进化完成，代数: {veb_evasion.generation}")

    # 测试RL更新
    stats = veb_evasion.update_rl()
    print(f"RL更新统计: {stats}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
