"""
个体编码与解码

每个个体是一个策略网络的权重
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class PolicyNetwork(nn.Module):
    """
    策略网络 - 将态势映射为动作

    Input: state_features (230维)
    Output: action_logits (40维 = 10单位 × 4动作维度)
    """

    def __init__(self, state_dim: int = 230, action_dim: int = 40,
                 hidden_dims: Tuple[int, ...] = (128, 64)):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # 构建网络层
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))

        self.net = nn.Sequential(*layers)

        # 计算总参数量
        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            state: 状态特征 shape=(batch, state_dim)

        Returns:
            action_logits: 动作logits shape=(batch, action_dim)
        """
        return self.net(state)

    def get_action(self, state: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        获取动作

        Args:
            state: 状态特征
            deterministic: 是否确定性选择

        Returns:
            action: 动作索引 shape=(batch, 10, 4) reshaped from (batch, 40)
        """
        logits = self.forward(state)  # (batch, 40)
        batch_size = logits.shape[0]

        # Reshape为每个单位的动作 (batch, 10, 4) -> 每单位4个动作维度
        # 动作维度: [方向8, 距离3, 速度3, 开火11]
        # 简化: 直接输出40维，外部解码
        if deterministic:
            # 对每个动作维度取argmax
            actions = []
            action_dims = [8, 3, 3, 11] * 10  # 10个单位
            offset = 0
            for dim in action_dims:
                action = logits[:, offset:offset+1].argmax(dim=-1) if dim == 1 else \
                         torch.zeros(batch_size, dtype=torch.long)
                offset += 1
            # 简化处理：直接返回logits用于外部解码
            return logits
        else:
            # 采样
            return logits


class Individual:
    """
    个体类 - 封装网络权重和适应度

    Attributes:
        weights: 扁平化的网络权重 np.array
        fitness: 适应度值
        q_value: Q网络评估值
        battle_score: 实战得分
    """

    def __init__(self, weights: np.ndarray = None, network: PolicyNetwork = None):
        """
        初始化个体

        Args:
            weights: 预设权重，None则随机初始化
            network: 用于推断权重维度的网络模板
        """
        if network is None:
            network = PolicyNetwork()

        self.num_params = network.num_params

        if weights is None:
            # 随机初始化 (Xavier-like)
            self.weights = np.random.randn(self.num_params).astype(np.float32) * 0.1
        else:
            self.weights = weights.astype(np.float32)

        # 适应度信息
        self.fitness = 0.0
        self.q_value = 0.0
        self.battle_score = 0.0
        self.wins = 0
        self.losses = 0

    def copy(self) -> 'Individual':
        """深拷贝"""
        new_ind = Individual(weights=self.weights.copy())
        new_ind.fitness = self.fitness
        new_ind.q_value = self.q_value
        new_ind.battle_score = self.battle_score
        return new_ind

    def load_into_network(self, network: PolicyNetwork):
        """将权重载入网络"""
        decode_weights(self.weights, network)

    def extract_from_network(self, network: PolicyNetwork):
        """从网络提取权重"""
        self.weights = encode_weights(network)


def encode_weights(network: PolicyNetwork) -> np.ndarray:
    """
    将网络权重编码为一维向量

    Args:
        network: 策略网络

    Returns:
        weights: 扁平化的权重数组
    """
    weights = []
    for param in network.parameters():
        weights.append(param.data.cpu().numpy().flatten())
    return np.concatenate(weights).astype(np.float32)


def decode_weights(weights: np.ndarray, network: PolicyNetwork):
    """
    将一维向量解码为网络权重

    Args:
        weights: 扁平化的权重数组
        network: 目标网络
    """
    offset = 0
    for param in network.parameters():
        size = param.numel()
        new_tensor = torch.from_numpy(
            weights[offset:offset + size].reshape(param.shape)
        ).float()
        param.data = new_tensor.to(param.device)
        offset += size


def create_random_individual(network_template: PolicyNetwork = None) -> Individual:
    """创建随机个体"""
    return Individual(network=network_template)


def create_individual_from_network(network: PolicyNetwork) -> Individual:
    """从网络创建个体"""
    ind = Individual(network=network)
    ind.extract_from_network(network)
    return ind


# 测试代码
if __name__ == "__main__":
    # 创建网络
    net = PolicyNetwork()
    print(f"Network parameters: {net.num_params}")

    # 创建个体
    ind = Individual(network=net)
    print(f"Individual weights shape: {ind.weights.shape}")

    # 测试编解码
    ind.load_into_network(net)
    state = torch.randn(4, 230)
    output = net(state)
    print(f"Output shape: {output.shape}")

    # 测试复制
    ind2 = ind.copy()
    print(f"Copy weights equal: {np.allclose(ind.weights, ind2.weights)}")
