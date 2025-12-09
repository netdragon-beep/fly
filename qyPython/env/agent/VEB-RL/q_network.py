# -*- coding: utf-8 -*-
"""
VEB-RL Q QܚI

: Value-Evolutionary-Based Reinforcement Learning (ICML 2024)

QQ(0-\<TD
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import copy


class QNetwork(nn.Module):
    """
    Q Q - 0-\< Q(s, a)

    ( VEB-RL -<0

    Input:
        state: y (state_dim)

    Output:
        q_values: *\Q< (action_dim)
    """

    def __init__(self, state_dim: int = 230, action_dim: int = 60,
                 hidden_dims: Tuple[int, ...] = (256, 256)):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # QB
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))

        self.net = nn.Sequential(*layers)

        # ;p
        self.num_params = sum(p.numel() for p in self.parameters())

        # C
        self._init_weights()

    def _init_weights(self):
        """Xavier """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        M 

        Args:
            state: y shape=(batch, state_dim)

        Returns:
            q_values: Q< shape=(batch, action_dim)
        """
        return self.net(state)

    def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
        """
        epsilon-greedy \	

        Args:
            state: y
            epsilon: "

        Returns:
            action: \"
        """
        if np.random.random() < epsilon:
            # :\
            batch_size = state.shape[0] if state.dim() > 1 else 1
            return torch.randint(0, self.action_dim, (batch_size,))
        else:
            # *j\
            with torch.no_grad():
                q_values = self.forward(state)
                return q_values.argmax(dim=-1)

    def get_max_q(self, state: torch.Tensor) -> torch.Tensor:
        """
        ' Q <

        Args:
            state: y

        Returns:
            max_q: 'Q<
        """
        with torch.no_grad():
            q_values = self.forward(state)
            return q_values.max(dim=-1)[0]


    def get_multi_agent_actions(
        self,
        state: torch.Tensor,
        num_agents: int = 5,
        actions_per_agent: int = 12,
        epsilon: float = 0.0
    ) -> torch.Tensor:
        """
        多智能体动作选择 - 每个智能体独立选择最优动作

        将 Q 值输出 (batch, num_agents * actions_per_agent) 重塑为
        (batch, num_agents, actions_per_agent)，每个智能体选择自己的最优动作

        Args:
            state: 状态 shape=(batch, state_dim) 或 (state_dim,)
            num_agents: 智能体数量（平台数）
            actions_per_agent: 每个智能体的动作数
            epsilon: 探索率

        Returns:
            actions: 每个智能体的动作 shape=(batch, num_agents) 或 (num_agents,)
        """
        single_input = state.dim() == 1
        if single_input:
            state = state.unsqueeze(0)

        batch_size = state.shape[0]

        with torch.no_grad():
            q_values = self.forward(state)  # (batch, 60)

            # 重塑为 (batch, num_agents, actions_per_agent)
            q_values = q_values.view(batch_size, num_agents, actions_per_agent)

            if np.random.random() < epsilon:
                # 探索：每个智能体随机选择动作
                actions = torch.randint(0, actions_per_agent, (batch_size, num_agents))
            else:
                # 利用：每个智能体选择自己Q值最大的动作
                actions = q_values.argmax(dim=-1)  # (batch, num_agents)

        if single_input:
            return actions.squeeze(0)  # (num_agents,)
        return actions


class DuelingQNetwork(nn.Module):
    """
    Dueling Q Q

    < V(s) p A(s, a)
    Q(s, a) = V(s) + A(s, a) - mean(A)
    """

    def __init__(self, state_dim: int = 230, action_dim: int = 60,
                 hidden_dims: Tuple[int, ...] = (256, 256)):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # qyB
        feature_layers = []
        prev_dim = state_dim

        for i, hidden_dim in enumerate(hidden_dims[:-1]):
            feature_layers.append(nn.Linear(prev_dim, hidden_dim))
            feature_layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.feature_net = nn.Sequential(*feature_layers) if feature_layers else nn.Identity()

        # 	qBprev_dim : state_dim
        if not feature_layers:
            prev_dim = state_dim

        # <A
        last_hidden = hidden_dims[-1] if hidden_dims else 256
        self.value_net = nn.Sequential(
            nn.Linear(prev_dim, last_hidden),
            nn.ReLU(),
            nn.Linear(last_hidden, 1)
        )

        # A
        self.advantage_net = nn.Sequential(
            nn.Linear(prev_dim, last_hidden),
            nn.ReLU(),
            nn.Linear(last_hidden, action_dim)
        )

        self.num_params = sum(p.numel() for p in self.parameters())
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_net(state)

        value = self.value_net(features)
        advantage = self.advantage_net(features)

        # Q = V + A - mean(A)
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values

    def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
        if np.random.random() < epsilon:
            batch_size = state.shape[0] if state.dim() > 1 else 1
            return torch.randint(0, self.action_dim, (batch_size,))
        else:
            with torch.no_grad():
                q_values = self.forward(state)
                return q_values.argmax(dim=-1)

    def get_max_q(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q_values = self.forward(state)
            return q_values.max(dim=-1)[0]



    def get_multi_agent_actions(
        self,
        state: torch.Tensor,
        num_agents: int = 5,
        actions_per_agent: int = 12,
        epsilon: float = 0.0
    ) -> torch.Tensor:
        """
        多智能体动作选择 - 每个智能体独立选择最优动作
        """
        single_input = state.dim() == 1
        if single_input:
            state = state.unsqueeze(0)

        batch_size = state.shape[0]

        with torch.no_grad():
            q_values = self.forward(state)
            q_values = q_values.view(batch_size, num_agents, actions_per_agent)

            if np.random.random() < epsilon:
                actions = torch.randint(0, actions_per_agent, (batch_size, num_agents))
            else:
                actions = q_values.argmax(dim=-1)

        if single_input:
            return actions.squeeze(0)
        return actions

def encode_q_weights(network: nn.Module) -> np.ndarray:
    """
    QQC:

    Args:
        network: QQ

    Returns:
        weights: AsCp
    """
    weights = []
    for param in network.parameters():
        weights.append(param.data.cpu().numpy().flatten())
    return np.concatenate(weights).astype(np.float32)


def decode_q_weights(weights: np.ndarray, network: nn.Module):
    """
    :QQC

    Args:
        weights: AsCp
        network: QQ
    """
    offset = 0
    for param in network.parameters():
        size = param.numel()
        new_tensor = torch.from_numpy(
            weights[offset:offset + size].reshape(param.shape)
        ).float()
        param.data = new_tensor.to(param.device)
        offset += size


def create_target_network(network: nn.Module) -> nn.Module:
    """
    Q	

    Args:
        network: QQ

    Returns:
        target_network: QQ
    """
    target = copy.deepcopy(network)
    # Qp
    for param in target.parameters():
        param.requires_grad = False
    return target


def soft_update(target: nn.Module, source: nn.Module, tau: float = 0.005):
    """
    oQ

    theta_target = tau * theta_source + (1 - tau) * theta_target

    Args:
        target: Q
        source: Q
        tau: op
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1 - tau) * target_param.data
        )


def hard_update(target: nn.Module, source: nn.Module):
    """
    lQ

    theta_target = theta_source
    """
    target.load_state_dict(source.state_dict())


# K
if __name__ == "__main__":
    # K Q Q
    q_net = QNetwork()
    print(f"QNetwork parameters: {q_net.num_params}")

    state = torch.randn(4, 230)
    q_values = q_net(state)
    print(f"Q values shape: {q_values.shape}")

    action = q_net.get_action(state, epsilon=0.1)
    print(f"Action shape: {action.shape}")

    # K Dueling Q Q
    dueling_net = DuelingQNetwork()
    print(f"\nDuelingQNetwork parameters: {dueling_net.num_params}")

    q_values = dueling_net(state)
    print(f"Dueling Q values shape: {q_values.shape}")

    # K
    weights = encode_q_weights(q_net)
    print(f"\nEncoded weights shape: {weights.shape}")

    q_net2 = QNetwork()
    decode_q_weights(weights, q_net2)

    # 
    q1 = q_net(state)
    q2 = q_net2(state)
    print(f"Weights equal: {torch.allclose(q1, q2)}")

    # KQ
    target_net = create_target_network(q_net)
    soft_update(target_net, q_net, tau=0.01)
    print(f"Target network created and updated")
