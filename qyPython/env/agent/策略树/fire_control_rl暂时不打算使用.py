"""
强化学习火控系统 (RL Fire Control)

基于SAC (Soft Actor-Critic) 算法的智能火控决策系统
- 与v1规则系统并行，可切换使用
- 通过在线训练学习最优开火策略
- 支持从历史数据离线预训练

作者: AI Assistant
日期: 2024
"""

import math
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

# 导入几何工具
try:
    from utilities.yxGeoUtils import YxGeoUtils
except ImportError:
    YxGeoUtils = None


# ==================== 配置类 ====================

@dataclass
class RLFireControlConfig:
    """RL火控配置"""
    # 状态空间维度
    state_dim: int = 9
    # 动作空间 (连续: 开火置信度 0-1)
    action_dim: int = 1
    # 隐藏层维度
    hidden_dim: int = 256
    # 学习率
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    # 折扣因子
    gamma: float = 0.99
    # 软更新系数
    tau: float = 0.005
    # 目标熵 (自动调节alpha)
    target_entropy: float = -1.0
    # 经验回放容量
    buffer_capacity: int = 100000
    # 批大小
    batch_size: int = 256
    # 开火阈值 (置信度超过此值才开火)
    fire_threshold: float = 0.5
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ==================== 神经网络模块 ====================

class Actor(nn.Module):
    """
    Actor网络 - 输出开火动作的均值和标准差

    输入: 状态向量 (batch, state_dim)
    输出: 动作均值和对数标准差 (batch, action_dim)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        # 初始化
        self._init_weights()

        # 动作范围限制
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样动作并计算对数概率

        返回: (action, log_prob)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # 重参数化采样
        normal = Normal(mean, std)
        x_t = normal.rsample()

        # Squash到[0,1]范围 (使用sigmoid)
        action = torch.sigmoid(x_t)

        # 计算对数概率 (考虑sigmoid变换的雅可比)
        log_prob = normal.log_prob(x_t)
        # sigmoid的雅可比修正
        log_prob -= torch.log(action * (1 - action) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        获取动作 (推理模式)
        """
        mean, log_std = self.forward(state)

        if deterministic:
            return torch.sigmoid(mean)
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()
            return torch.sigmoid(x_t)


class Critic(nn.Module):
    """
    Critic网络 - 双Q网络结构

    输入: 状态 + 动作
    输出: Q值
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()

        # Q1网络
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)

        # Q2网络
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)

        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)

        q2 = F.relu(self.q2_fc1(x))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_out(q2)

        return q1, q2

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        return self.q1_out(q1)


# ==================== 经验回放 ====================

class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int, state_dim: int, action_dim: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        # 预分配内存
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool):
        """添加经验"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """采样批次"""
        idx = np.random.randint(0, self.size, size=batch_size)

        return {
            'states': torch.FloatTensor(self.states[idx]).to(self.device),
            'actions': torch.FloatTensor(self.actions[idx]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[idx]).to(self.device),
            'next_states': torch.FloatTensor(self.next_states[idx]).to(self.device),
            'dones': torch.FloatTensor(self.dones[idx]).to(self.device)
        }

    def __len__(self) -> int:
        return self.size


# ==================== SAC智能体 ====================

class SACFireControl:
    """
    基于SAC的智能火控系统

    使用方法:
    1. 训练模式: 调用 update() 更新网络
    2. 推理模式: 调用 should_fire() 获取开火决策
    """

    def __init__(self, config: RLFireControlConfig = None):
        self.config = config or RLFireControlConfig()
        self.device = torch.device(self.config.device)

        # 创建网络
        self.actor = Actor(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dim
        ).to(self.device)

        self.critic = Critic(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dim
        ).to(self.device)

        self.critic_target = Critic(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dim
        ).to(self.device)

        # 复制参数到目标网络
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)

        # 自动调节的温度参数alpha
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.alpha_lr)
        self.target_entropy = self.config.target_entropy

        # 经验回放
        self.replay_buffer = ReplayBuffer(
            self.config.buffer_capacity,
            self.config.state_dim,
            self.config.action_dim,
            self.config.device
        )

        # 训练统计
        self.train_steps = 0
        self.episode_rewards = []

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ==================== 状态提取 ====================

    def extract_state(self, shooter: Dict, target: Dict) -> np.ndarray:
        """
        从射手和目标信息提取状态向量

        状态空间 (9维):
        0. 归一化距离 (0-1, 基于最大射程20km)
        1. 归一化姿态角 (0-1, 0-180°)
        2. 归一化接近率 (0-1, -500~700 m/s)
        3. 是否在NEZ内 (0/1)
        4. 归一化射手速度 (0-1, 0-500 m/s)
        5. 归一化目标速度 (0-1, 0-500 m/s)
        6. 归一化高度差 (0-1, -5000~5000 m)
        7. 目标是否是有人机 (0/1)
        8. 射手是否是有人机 (0/1)
        """
        # 获取射手信息
        s_lon = shooter.get('longitude', 0)
        s_lat = shooter.get('latitude', 0)
        s_speed = shooter.get('speed', 300)
        s_heading = shooter.get('heading', 0)
        s_alt = shooter.get('altitude', 5000)
        is_manned_shooter = shooter.get('type') == '有人机'

        # 获取目标信息
        t_lon = target.get('longitude', target.get('X', 0))
        t_lat = target.get('latitude', target.get('Y', 0))
        t_speed = target.get('speed', 300)
        t_heading = target.get('heading', 0)
        t_alt = target.get('altitude', target.get('Alt', 5000))
        is_manned_target = target.get('platform_entity_type') == '有人机'

        # 计算距离
        if YxGeoUtils:
            distance = YxGeoUtils.haversine_distance(s_lon, s_lat, t_lon, t_lat)
        else:
            # 简化计算
            distance = math.sqrt((s_lon - t_lon)**2 + (s_lat - t_lat)**2) * 111000

        # 计算姿态角
        aspect_angle = self._calculate_aspect_angle(
            s_lon, s_lat, s_heading,
            t_lon, t_lat, t_heading
        )

        # 计算接近率
        closure_rate = self._calculate_closure_rate(
            s_lon, s_lat, s_speed, s_heading,
            t_lon, t_lat, t_speed, t_heading
        )

        # 计算NEZ
        max_range = 20000 if is_manned_shooter else 15000
        nez_head = 12000 if is_manned_shooter else 8000
        nez_tail = 6000 if is_manned_shooter else 4000
        nez = nez_head - (nez_head - nez_tail) * (aspect_angle / 180.0)
        in_nez = 1.0 if distance <= nez else 0.0

        # 高度差
        alt_diff = s_alt - t_alt

        # 构建状态向量 (归一化)
        state = np.array([
            min(distance / max_range, 1.0),           # 距离
            aspect_angle / 180.0,                      # 姿态角
            (closure_rate + 500) / 1200.0,            # 接近率 (偏移到正数范围)
            in_nez,                                    # NEZ内
            min(s_speed / 500.0, 1.0),                # 射手速度
            min(t_speed / 500.0, 1.0),                # 目标速度
            (alt_diff + 5000) / 10000.0,              # 高度差
            1.0 if is_manned_target else 0.0,         # 目标类型
            1.0 if is_manned_shooter else 0.0         # 射手类型
        ], dtype=np.float32)

        return state

    def _calculate_aspect_angle(self, s_lon, s_lat, s_heading, t_lon, t_lat, t_heading) -> float:
        """计算姿态角"""
        if YxGeoUtils:
            bearing_to_shooter = YxGeoUtils.calculate_bearing(t_lon, t_lat, s_lon, s_lat)
        else:
            # 简化计算
            dx = s_lon - t_lon
            dy = s_lat - t_lat
            bearing_to_shooter = math.degrees(math.atan2(dx, dy)) % 360

        target_hdg = math.degrees(t_heading) % 360 if isinstance(t_heading, float) else t_heading % 360
        aspect = abs((bearing_to_shooter - target_hdg + 180) % 360 - 180)
        return aspect

    def _calculate_closure_rate(self, s_lon, s_lat, s_speed, s_heading,
                                 t_lon, t_lat, t_speed, t_heading) -> float:
        """计算接近率"""
        if YxGeoUtils:
            bearing_to_target = YxGeoUtils.calculate_bearing(s_lon, s_lat, t_lon, t_lat)
        else:
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

    # ==================== 推理接口 ====================

    def should_fire(self, shooter: Dict, target: Dict,
                    deterministic: bool = True) -> Tuple[bool, float, str]:
        """
        判断是否应该开火 (推理模式)

        Args:
            shooter: 射手信息
            target: 目标信息
            deterministic: 是否确定性策略

        Returns:
            (should_fire, confidence, reason)
        """
        # 提取状态
        state = self.extract_state(shooter, target)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # 获取动作 (开火置信度)
        with torch.no_grad():
            confidence = self.actor.get_action(state_tensor, deterministic=deterministic)
            confidence = confidence.cpu().numpy()[0, 0]

        # 决策
        should_fire = confidence >= self.config.fire_threshold

        if should_fire:
            reason = f"RL开火(conf={confidence:.2f})"
        else:
            reason = f"RL不开火(conf={confidence:.2f}<{self.config.fire_threshold})"

        return should_fire, confidence, reason

    def batch_should_fire(self, pairs: List[Tuple[Dict, Dict]],
                          deterministic: bool = True) -> List[Tuple[bool, float, str]]:
        """
        批量判断是否应该开火 (高效版本)

        Args:
            pairs: [(shooter1, target1), (shooter2, target2), ...] 射手-目标对列表
            deterministic: 是否确定性策略

        Returns:
            [(should_fire, confidence, reason), ...] 结果列表
        """
        if not pairs:
            return []

        # 批量提取状态
        states = np.array([self.extract_state(s, t) for s, t in pairs], dtype=np.float32)
        states_tensor = torch.FloatTensor(states).to(self.device)

        # 一次性推理所有
        with torch.no_grad():
            confidences = self.actor.get_action(states_tensor, deterministic=deterministic)
            confidences = confidences.cpu().numpy().flatten()

        # 生成结果
        results = []
        for conf in confidences:
            should_fire = conf >= self.config.fire_threshold
            if should_fire:
                reason = f"RL开火(conf={conf:.2f})"
            else:
                reason = f"RL不开火(conf={conf:.2f}<{self.config.fire_threshold})"
            results.append((should_fire, float(conf), reason))

        return results

    # ==================== 训练接口 ====================

    def store_transition(self, shooter: Dict, target: Dict, action: float,
                         reward: float, next_shooter: Dict, next_target: Dict, done: bool):
        """存储经验"""
        state = self.extract_state(shooter, target)
        next_state = self.extract_state(next_shooter, next_target)
        self.replay_buffer.add(state, np.array([action]), reward, next_state, done)

    def store_transition_raw(self, state: np.ndarray, action: float, reward: float,
                              next_state: np.ndarray, done: bool):
        """直接存储原始状态经验"""
        self.replay_buffer.add(state, np.array([action]), reward, next_state, done)

    def update(self) -> Dict[str, float]:
        """
        SAC更新步骤

        Returns:
            训练统计信息
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return {}

        # 采样
        batch = self.replay_buffer.sample(self.config.batch_size)
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']

        # ========== 更新Critic ==========
        with torch.no_grad():
            # 采样下一步动作
            next_actions, next_log_probs = self.actor.sample(next_states)

            # 目标Q值
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target)

            # TD目标 (带熵正则)
            td_target = rewards + self.config.gamma * (1 - dones) * (q_target - self.alpha * next_log_probs)

        # 当前Q值
        q1, q2 = self.critic(states, actions)

        # Critic损失
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ========== 更新Actor ==========
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        # Actor损失 (最大化 Q - alpha * log_prob)
        actor_loss = (self.alpha.detach() * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ========== 更新Alpha ==========
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ========== 软更新目标网络 ==========
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

        self.train_steps += 1

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item(),
            'q_value': q_new.mean().item()
        }

    # ==================== 模型保存/加载 ====================

    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'train_steps': self.train_steps,
            'config': self.config
        }, filepath)
        print(f"[SAC] 模型已保存到: {filepath}")

    def load(self, filepath: str):
        """加载模型"""
        # weights_only=False 因为我们保存了自定义config对象
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        self.train_steps = checkpoint['train_steps']

        print(f"[SAC] 模型已加载: {filepath}, 训练步数: {self.train_steps}")

    # ==================== 从历史数据预训练 ====================

    def pretrain_from_data(self, data_path: str, epochs: int = 100):
        """
        从历史击杀数据预训练

        Args:
            data_path: kill_data.json 路径
            epochs: 训练轮数
        """
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            records = json.load(f)

        print(f"[SAC] 加载 {len(records)} 条历史数据进行预训练")

        # 转换为状态-动作-奖励
        for record in records:
            # 构造状态
            max_range = 20000 if record.get('shooter_type') == '有人机' else 15000
            nez_head = 12000 if record.get('shooter_type') == '有人机' else 8000
            nez_tail = 6000 if record.get('shooter_type') == '有人机' else 4000
            aspect_angle = record.get('aspect_angle', 90)
            nez = nez_head - (nez_head - nez_tail) * (aspect_angle / 180.0)

            state = np.array([
                min(record.get('distance', 10000) / max_range, 1.0),
                record.get('aspect_angle', 90) / 180.0,
                (record.get('closure_rate', 200) + 500) / 1200.0,
                1.0 if record.get('distance', 10000) <= nez else 0.0,
                min(record.get('shooter_speed', 300) / 500.0, 1.0),
                min(record.get('target_speed', 300) / 500.0, 1.0),
                (record.get('altitude_diff', 0) + 5000) / 10000.0,
                1.0 if record.get('target_type') == '有人机' else 0.0,
                1.0 if record.get('shooter_type') == '有人机' else 0.0
            ], dtype=np.float32)

            # 动作 (开火了所以action=1)
            action = 1.0

            # 奖励
            if record.get('result') == 'kill':
                reward = 1.0
            else:
                reward = -1.0

            # 存储 (next_state简化为same state, done=True)
            self.replay_buffer.add(state, np.array([action]), reward, state, True)

        # 训练
        print(f"[SAC] 开始预训练 {epochs} 轮...")
        for epoch in range(epochs):
            stats = self.update()
            if (epoch + 1) % 10 == 0 and stats:
                print(f"  Epoch {epoch+1}/{epochs}: "
                      f"critic_loss={stats['critic_loss']:.4f}, "
                      f"actor_loss={stats['actor_loss']:.4f}, "
                      f"alpha={stats['alpha']:.4f}")

        print(f"[SAC] 预训练完成!")


# ==================== 混合火控系统 ====================

class HybridFireControl:
    """
    混合火控系统 - 可在规则和RL之间切换

    使用方法:
    1. set_mode('rule') - 使用v1规则系统
    2. set_mode('rl') - 使用SAC强化学习
    3. set_mode('hybrid') - 混合决策
    """

    def __init__(self, rl_model_path: str = None):
        self.mode = 'rule'  # 默认使用规则

        # 初始化RL系统
        self.rl_fire_control = SACFireControl()
        if rl_model_path and os.path.exists(rl_model_path):
            self.rl_fire_control.load(rl_model_path)

        # 导入规则系统
        try:
            from fire_control import SmartFireControl
            self.rule_fire_control = SmartFireControl
        except ImportError:
            self.rule_fire_control = None

    def set_mode(self, mode: str):
        """设置模式: 'rule', 'rl', 'hybrid'"""
        assert mode in ['rule', 'rl', 'hybrid']
        self.mode = mode
        print(f"[HybridFC] 切换到 {mode} 模式")

    def should_fire(self, shooter: Dict, target: Dict, agent=None) -> Tuple[bool, float, str]:
        """
        判断是否应该开火
        """
        if self.mode == 'rule':
            if self.rule_fire_control:
                return self.rule_fire_control.should_fire(shooter, target, agent)
            else:
                return False, 0.0, "规则系统不可用"

        elif self.mode == 'rl':
            return self.rl_fire_control.should_fire(shooter, target)

        elif self.mode == 'hybrid':
            # 混合模式: 两者都同意才开火
            rule_result = self.rule_fire_control.should_fire(shooter, target, agent) if self.rule_fire_control else (False, 0.0, "")
            rl_result = self.rl_fire_control.should_fire(shooter, target)

            rule_fire, rule_pk, rule_reason = rule_result
            rl_fire, rl_conf, rl_reason = rl_result

            # 两者都同意才开火
            should_fire = rule_fire and rl_fire
            confidence = (rule_pk + rl_conf) / 2
            reason = f"混合({rule_reason}+{rl_reason})"

            return should_fire, confidence, reason

        return False, 0.0, "未知模式"

    def batch_should_fire(self, pairs: List[Tuple[Dict, Dict]], agent=None) -> List[Tuple[bool, float, str]]:
        """
        批量判断是否应该开火 (高效版本)
        """
        if not pairs:
            return []

        if self.mode == 'rl':
            return self.rl_fire_control.batch_should_fire(pairs)
        else:
            # 规则模式或混合模式，逐个调用
            return [self.should_fire(s, t, agent) for s, t in pairs]


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("SAC火控系统测试")
    print("=" * 60)

    # 创建SAC火控
    config = RLFireControlConfig(device="cpu")
    sac = SACFireControl(config)

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
    state = sac.extract_state(shooter, target)
    print(f"\n状态向量 (9维):")
    print(f"  距离: {state[0]:.3f}")
    print(f"  姿态角: {state[1]:.3f}")
    print(f"  接近率: {state[2]:.3f}")
    print(f"  NEZ内: {state[3]:.1f}")
    print(f"  射手速度: {state[4]:.3f}")
    print(f"  目标速度: {state[5]:.3f}")
    print(f"  高度差: {state[6]:.3f}")
    print(f"  目标有人机: {state[7]:.1f}")
    print(f"  射手有人机: {state[8]:.1f}")

    # 测试推理
    should_fire, confidence, reason = sac.should_fire(shooter, target)
    print(f"\n开火决策:")
    print(f"  应该开火: {should_fire}")
    print(f"  置信度: {confidence:.3f}")
    print(f"  原因: {reason}")

    # 测试预训练 (如果有数据)
    data_path = os.path.join(os.path.dirname(__file__), 'kill_data.json')
    if os.path.exists(data_path):
        print(f"\n发现历史数据，开始预训练...")
        sac.pretrain_from_data(data_path, epochs=50)

        # 再次测试
        should_fire, confidence, reason = sac.should_fire(shooter, target)
        print(f"\n预训练后开火决策:")
        print(f"  应该开火: {should_fire}")
        print(f"  置信度: {confidence:.3f}")
        print(f"  原因: {reason}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
