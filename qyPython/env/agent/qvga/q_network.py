"""
Q网络 - 评估策略价值

输入: 状态特征 + 策略嵌入
输出: Q值 (预期累积奖励)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional
from collections import deque
import random


class QNetwork(nn.Module):
    """
    Q网络 - 评估策略在给定状态下的价值

    可以有两种模式:
    1. Q(s, policy_embedding) - 评估特定策略
    2. Q(s) - 评估状态价值 (简化版)
    """

    def __init__(self, state_dim: int = 230, policy_dim: int = 64,
                 hidden_dim: int = 256, use_policy_embedding: bool = True):
        super().__init__()

        self.state_dim = state_dim
        self.policy_dim = policy_dim
        self.use_policy_embedding = use_policy_embedding

        if use_policy_embedding:
            input_dim = state_dim + policy_dim
        else:
            input_dim = state_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 策略编码器 (将策略权重压缩为低维嵌入)
        if use_policy_embedding:
            self.policy_encoder = nn.Sequential(
                nn.Linear(40000, 512),  # 假设策略约40000参数
                nn.ReLU(),
                nn.Linear(512, policy_dim)
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor,
                policy_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            state: 状态特征 shape=(batch, state_dim)
            policy_weights: 策略权重 shape=(batch, policy_params) 可选

        Returns:
            q_value: Q值 shape=(batch, 1)
        """
        if self.use_policy_embedding and policy_weights is not None:
            # 编码策略
            policy_embed = self.policy_encoder(policy_weights)
            x = torch.cat([state, policy_embed], dim=-1)
        else:
            x = state

        return self.net(x)


class SimpleQNetwork(nn.Module):
    """
    简化版Q网络 - 只评估状态价值

    不需要策略嵌入，适用于:
    - 评估当前态势的整体价值
    - 作为baseline比较
    """

    def __init__(self, state_dim: int = 230, hidden_dim: int = 256, use_policy_embedding: bool = False):
        super().__init__()

        self.state_dim = state_dim
        self.use_policy_embedding = False  # SimpleQNetwork 始终不使用策略嵌入

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state: np.ndarray, policy_weights: np.ndarray,
            reward: float, next_state: np.ndarray, done: bool):
        """添加经验"""
        self.buffer.append((state, policy_weights, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """随机采样"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        states = np.array([t[0] for t in batch])
        policies = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])

        return states, policies, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class QNetworkTrainer:
    """Q网络训练器"""

    def __init__(self, q_network: nn.Module, lr: float = 1e-3,
                 gamma: float = 0.99, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.q_network = q_network.to(self.device)
        self.target_network = type(q_network)(
            state_dim=q_network.state_dim,
            hidden_dim=256,
            use_policy_embedding=q_network.use_policy_embedding
        ).to(self.device)
        self.target_network.load_state_dict(q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.gamma = gamma

        self.buffer = ReplayBuffer()
        self.update_count = 0
        self.target_update_freq = 100

    def add_experience(self, state: np.ndarray, policy_weights: np.ndarray,
                       reward: float, next_state: np.ndarray, done: bool):
        """添加经验"""
        self.buffer.add(state, policy_weights, reward, next_state, done)

    def update(self, batch_size: int = 64) -> float:
        """更新Q网络"""
        if len(self.buffer) < batch_size:
            return 0.0

        # 采样
        states, policies, rewards, next_states, dones = self.buffer.sample(batch_size)

        # 转换为张量
        states_t = torch.FloatTensor(states).to(self.device)
        policies_t = torch.FloatTensor(policies).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # 计算当前Q值
        if self.q_network.use_policy_embedding:
            current_q = self.q_network(states_t, policies_t).squeeze()
        else:
            current_q = self.q_network(states_t).squeeze()

        # 计算目标Q值
        with torch.no_grad():
            if self.target_network.use_policy_embedding:
                next_q = self.target_network(next_states_t, policies_t).squeeze()
            else:
                next_q = self.target_network(next_states_t).squeeze()
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        # 计算损失
        loss = nn.functional.mse_loss(current_q, target_q)

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # 更新目标网络
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def evaluate(self, state: np.ndarray,
                 policy_weights: np.ndarray = None) -> float:
        """评估Q值"""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if policy_weights is not None and self.q_network.use_policy_embedding:
                policy_t = torch.FloatTensor(policy_weights).unsqueeze(0).to(self.device)
                q_value = self.q_network(state_t, policy_t).item()
            else:
                q_value = self.q_network(state_t).item()
        return q_value

    def batch_evaluate(self, states: np.ndarray,
                       policy_weights: np.ndarray = None) -> np.ndarray:
        """批量评估Q值"""
        with torch.no_grad():
            states_t = torch.FloatTensor(states).to(self.device)
            if policy_weights is not None and self.q_network.use_policy_embedding:
                policies_t = torch.FloatTensor(policy_weights).to(self.device)
                q_values = self.q_network(states_t, policies_t).squeeze().cpu().numpy()
            else:
                q_values = self.q_network(states_t).squeeze().cpu().numpy()
        return q_values

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


# 测试代码
if __name__ == "__main__":
    # 测试简化版Q网络
    simple_q = SimpleQNetwork()
    state = torch.randn(4, 230)
    q_value = simple_q(state)
    print(f"Simple Q output shape: {q_value.shape}")

    # 测试完整版Q网络
    full_q = QNetwork(use_policy_embedding=False)  # 先测试不用策略嵌入
    q_value = full_q(state)
    print(f"Full Q output shape: {q_value.shape}")

    # 测试训练器
    trainer = QNetworkTrainer(full_q, device='cpu')
    for _ in range(100):
        trainer.add_experience(
            state=np.random.randn(230).astype(np.float32),
            policy_weights=np.random.randn(40000).astype(np.float32),
            reward=np.random.randn(),
            next_state=np.random.randn(230).astype(np.float32),
            done=False
        )

    loss = trainer.update()
    print(f"Training loss: {loss}")
