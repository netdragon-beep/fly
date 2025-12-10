"""
Q网络模块

用于VEB-RL的价值函数网络
支持标准DQN和Dueling DQN架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class QNetworkConfig:
    """Q网络配置"""
    state_dim: int = 9          # 状态维度
    action_dim: int = 2         # 动作维度 (开火/不开火)
    hidden_dims: Tuple[int, ...] = (256, 128)  # 隐藏层维度
    dueling: bool = True        # 是否使用Dueling架构
    noisy: bool = False         # 是否使用NoisyNet


class NoisyLinear(nn.Module):
    """
    NoisyNet线性层 - 用于探索
    参考: Fortunato et al., 2018
    """
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # 可学习参数
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # 噪声缓冲
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def reset_noise(self):
        """重置噪声"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class QNetwork(nn.Module):
    """
    标准Q网络

    输入: 状态向量 (batch, state_dim)
    输出: Q值 (batch, action_dim)
    """

    def __init__(self, config: QNetworkConfig):
        super().__init__()
        self.config = config

        # 构建网络层
        layers = []
        in_dim = config.state_dim

        for hidden_dim in config.hidden_dims:
            if config.noisy:
                layers.append(NoisyLinear(in_dim, hidden_dim))
            else:
                layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        self.features = nn.Sequential(*layers)

        # 输出层
        if config.noisy:
            self.output = NoisyLinear(in_dim, config.action_dim)
        else:
            self.output = nn.Linear(in_dim, config.action_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.features(state)
        return self.output(features)

    def reset_noise(self):
        """重置所有NoisyLinear层的噪声"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class DuelingQNetwork(nn.Module):
    """
    Dueling Q网络

    将Q值分解为: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
    参考: Wang et al., 2016
    """

    def __init__(self, config: QNetworkConfig):
        super().__init__()
        self.config = config

        # 共享特征提取
        layers = []
        in_dim = config.state_dim

        for hidden_dim in config.hidden_dims[:-1]:
            if config.noisy:
                layers.append(NoisyLinear(in_dim, hidden_dim))
            else:
                layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        self.features = nn.Sequential(*layers) if layers else nn.Identity()

        # 最后一个隐藏层维度
        last_hidden = config.hidden_dims[-1] if config.hidden_dims else config.state_dim

        # 价值流 (Value stream)
        if config.noisy:
            self.value_hidden = NoisyLinear(in_dim, last_hidden)
            self.value = NoisyLinear(last_hidden, 1)
        else:
            self.value_hidden = nn.Linear(in_dim, last_hidden)
            self.value = nn.Linear(last_hidden, 1)

        # 优势流 (Advantage stream)
        if config.noisy:
            self.advantage_hidden = NoisyLinear(in_dim, last_hidden)
            self.advantage = NoisyLinear(last_hidden, config.action_dim)
        else:
            self.advantage_hidden = nn.Linear(in_dim, last_hidden)
            self.advantage = nn.Linear(last_hidden, config.action_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.features(state)

        # 价值流
        value = F.relu(self.value_hidden(features))
        value = self.value(value)

        # 优势流
        advantage = F.relu(self.advantage_hidden(features))
        advantage = self.advantage(advantage)

        # 组合: Q = V + A - mean(A)
        q_value = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_value

    def reset_noise(self):
        """重置所有NoisyLinear层的噪声"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# ==================== 工具函数 ====================

def create_q_network(config: QNetworkConfig, device: str = 'cpu') -> nn.Module:
    """创建Q网络"""
    if config.dueling:
        net = DuelingQNetwork(config)
    else:
        net = QNetwork(config)
    return net.to(device)


def hard_update(source: nn.Module, target: nn.Module):
    """硬更新: 完全复制参数"""
    target.load_state_dict(source.state_dict())


def soft_update(source: nn.Module, target: nn.Module, tau: float = 0.005):
    """软更新: target = tau * source + (1-tau) * target"""
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Q网络模块测试")
    print("=" * 60)

    # 测试配置
    config = QNetworkConfig(
        state_dim=9,
        action_dim=2,
        hidden_dims=(256, 128),
        dueling=True,
        noisy=False
    )

    # 创建网络
    q_net = create_q_network(config)
    print(f"网络结构:\n{q_net}")

    # 测试前向传播
    batch_size = 32
    state = torch.randn(batch_size, config.state_dim)
    q_values = q_net(state)

    print(f"\n输入形状: {state.shape}")
    print(f"输出形状: {q_values.shape}")
    print(f"Q值范例: {q_values[0]}")

    # 测试参数数量
    total_params = sum(p.numel() for p in q_net.parameters())
    print(f"\n总参数量: {total_params:,}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
