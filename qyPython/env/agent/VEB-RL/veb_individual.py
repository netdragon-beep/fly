# -*- coding: utf-8 -*-
"""
VEB-RL *SšI

úŽº‡: Value-Evolutionary-Based Reinforcement Learning (ICML 2024)

Ï**S+:
- QQÜ (theta)
- îQQÜ (theta')
- ”¦ = TDïî: f(theta, theta') = -E[(r + gamma * max_a' Q_theta'(s', a') - Q_theta(s, a))^2]
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List
import copy

from .q_network import (
    QNetwork, DuelingQNetwork,
    encode_q_weights, decode_q_weights,
    create_target_network, hard_update
)


class VEBIndividual:
    """
    VEB-RL *S{

    + Q QÜŒî Q QÜ”¦úŽ TD ïî

    Attributes:
        q_weights: QQÜ„AsCÍ
        target_weights: îQQÜ„AsCÍ
        fitness: ”¦< (TDïî)
        td_error: TDïî
        episode_return: /ïÞ¥(Žå×	
    """

    def __init__(
        self,
        q_weights: np.ndarray = None,
        target_weights: np.ndarray = None,
        state_dim: int = 230,
        action_dim: int = 40,
        hidden_dims: Tuple[int, ...] = (256, 256),
        use_dueling: bool = False
    ):
        """
        Ë*S

        Args:
            q_weights: QQÜCÍNone:Ë
            target_weights: îQÜCÍNone6QQÜ
            state_dim: ¶ô¦
            action_dim: ¨\ô¦
            hidden_dims: ÏBô¦
            use_dueling: /&(DuelingQÜ
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.use_dueling = use_dueling

        # ú!QÜ·ÖÂppÏ
        if use_dueling:
            template = DuelingQNetwork(state_dim, action_dim, hidden_dims)
        else:
            template = QNetwork(state_dim, action_dim, hidden_dims)

        self.num_params = template.num_params

        # Ë Q QÜCÍ
        if q_weights is None:
            self.q_weights = encode_q_weights(template)
        else:
            self.q_weights = q_weights.astype(np.float32)

        # ËîQÜCÍ
        if target_weights is None:
            self.target_weights = self.q_weights.copy()
        else:
            self.target_weights = target_weights.astype(np.float32)

        # ”¦Œß¡áo
        self.fitness = -float('inf')  # TDïîŠ'Š}
        self.td_error = float('inf')
        self.episode_return = 0.0
        self.episodes_evaluated = 0

    def create_q_network(self, device: str = 'cpu') -> nn.Module:
        """úQQÜv }CÍ"""
        if self.use_dueling:
            network = DuelingQNetwork(
                self.state_dim, self.action_dim, self.hidden_dims
            ).to(device)
        else:
            network = QNetwork(
                self.state_dim, self.action_dim, self.hidden_dims
            ).to(device)

        decode_q_weights(self.q_weights, network)
        return network

    def create_target_network(self, device: str = 'cpu') -> nn.Module:
        """úîQQÜv }CÍ"""
        if self.use_dueling:
            network = DuelingQNetwork(
                self.state_dim, self.action_dim, self.hidden_dims
            ).to(device)
        else:
            network = QNetwork(
                self.state_dim, self.action_dim, self.hidden_dims
            ).to(device)

        decode_q_weights(self.target_weights, network)
        # »ÓÂp
        for param in network.parameters():
            param.requires_grad = False
        return network

    def update_from_network(self, q_network: nn.Module):
        """ÎQQÜô°CÍ"""
        self.q_weights = encode_q_weights(q_network)

    def update_target_from_q(self):
        """QQÜCÍ60îQÜ"""
        self.target_weights = self.q_weights.copy()

    def compute_fitness(
        self,
        replay_buffer: 'ReplayBuffer',
        batch_size: int = 256,
        gamma: float = 0.99,
        device: str = 'cpu',
        n_batches: int = 10
    ) -> float:
        """
        ¡—”¦TDïî	

        f(theta, theta') = -E[(r + gamma * max_a' Q_theta'(s', a') - Q_theta(s, a))^2]

        Args:
            replay_buffer: ÏŒÞ>²:
            batch_size: yÏ'
            gamma: ˜càP
            device: ¡—¾
            n_batches: (ŽÄ0„y!p

        Returns:
            fitness: ”¦<
        """
        if len(replay_buffer) < batch_size:
            return -float('inf')

        q_net = self.create_q_network(device)
        target_net = self.create_target_network(device)

        q_net.eval()
        target_net.eval()

        total_td_error = 0.0

        with torch.no_grad():
            for _ in range(n_batches):
                batch = replay_buffer.sample(batch_size)

                states = torch.FloatTensor(batch['states']).to(device)
                actions = torch.LongTensor(batch['actions']).to(device)
                rewards = torch.FloatTensor(batch['rewards']).to(device)
                next_states = torch.FloatTensor(batch['next_states']).to(device)
                dones = torch.FloatTensor(batch['dones']).to(device)

                # Q(s, a)
                q_values = q_net(states)
                q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                # max_a' Q_theta'(s', a')
                next_q_values = target_net(next_states)
                max_next_q = next_q_values.max(dim=1)[0]

                # TD target: r + gamma * max_a' Q_theta'(s', a') * (1 - done)
                td_target = rewards + gamma * max_next_q * (1 - dones)

                # TD error: (td_target - q_sa)^2
                td_error = (td_target - q_sa).pow(2).mean().item()
                total_td_error += td_error

        avg_td_error = total_td_error / n_batches
        self.td_error = avg_td_error
        self.fitness = -avg_td_error  # TDïî\:”¦

        return self.fitness

    def copy(self) -> 'VEBIndividual':
        """ñ÷"""
        new_ind = VEBIndividual(
            q_weights=self.q_weights.copy(),
            target_weights=self.target_weights.copy(),
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=self.hidden_dims,
            use_dueling=self.use_dueling
        )
        new_ind.fitness = self.fitness
        new_ind.td_error = self.td_error
        new_ind.episode_return = self.episode_return
        return new_ind

    def __repr__(self):
        return (f"VEBIndividual(fitness={self.fitness:.4f}, "
                f"td_error={self.td_error:.4f}, "
                f"params={self.num_params})")


class ReplayBuffer:
    """
    ÏŒÞ>²:

    X¨ (s, a, r, s', done) lû
    """

    def __init__(self, capacity: int = 100000, state_dim: int = 230):
        self.capacity = capacity
        self.state_dim = state_dim

        # „M…X
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool):
        """û lû"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, states: np.ndarray, actions: np.ndarray,
                  rewards: np.ndarray, next_states: np.ndarray,
                  dones: np.ndarray):
        """yÏû lû"""
        batch_size = len(states)
        indices = np.arange(self.ptr, self.ptr + batch_size) % self.capacity

        self.states[indices] = states
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.next_states[indices] = next_states
        self.dones[indices] = dones

        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int) -> dict:
        """:Ç7yÏ"""
        indices = np.random.randint(0, self.size, size=batch_size)

        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices]
        }

    def __len__(self):
        return self.size

    def clear(self):
        """z²:"""
        self.ptr = 0
        self.size = 0


class VEBPopulation:
    """
    VEB-RL Í¤

    ¡ Ä VEBIndividual
    """

    def __init__(
        self,
        size: int = 50,
        state_dim: int = 230,
        action_dim: int = 40,
        hidden_dims: Tuple[int, ...] = (256, 256),
        use_dueling: bool = False
    ):
        self.size = size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.use_dueling = use_dueling

        self.individuals: List[VEBIndividual] = []
        self.generation = 0

        # ß¡áo
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_td_error': [],
            'avg_td_error': [],
            'best_return': [],
            'avg_return': []
        }

    def initialize_random(self):
        """:ËÍ¤"""
        self.individuals = [
            VEBIndividual(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dims=self.hidden_dims,
                use_dueling=self.use_dueling
            )
            for _ in range(self.size)
        ]
        self.generation = 0

    def get_elite(self, n: int = 1) -> List[VEBIndividual]:
        """·Ö”¦ Ø„n**S"""
        sorted_inds = sorted(
            self.individuals,
            key=lambda x: x.fitness,
            reverse=True
        )
        return sorted_inds[:n]

    def get_worst(self, n: int = 1) -> List[VEBIndividual]:
        """·Ö”¦ N„n**S"""
        sorted_inds = sorted(
            self.individuals,
            key=lambda x: x.fitness
        )
        return sorted_inds[:n]

    def compute_statistics(self) -> dict:
        """¡—Í¤ß¡áo"""
        fitnesses = [ind.fitness for ind in self.individuals
                     if ind.fitness > -float('inf')]
        td_errors = [ind.td_error for ind in self.individuals
                     if ind.td_error < float('inf')]
        returns = [ind.episode_return for ind in self.individuals]

        stats = {}

        if fitnesses:
            stats['best_fitness'] = max(fitnesses)
            stats['avg_fitness'] = np.mean(fitnesses)
            stats['worst_fitness'] = min(fitnesses)
        else:
            stats['best_fitness'] = -float('inf')
            stats['avg_fitness'] = -float('inf')
            stats['worst_fitness'] = -float('inf')

        if td_errors:
            stats['best_td_error'] = min(td_errors)
            stats['avg_td_error'] = np.mean(td_errors)
        else:
            stats['best_td_error'] = float('inf')
            stats['avg_td_error'] = float('inf')

        stats['best_return'] = max(returns) if returns else 0.0
        stats['avg_return'] = np.mean(returns) if returns else 0.0

        # °U†ò
        self.history['best_fitness'].append(stats['best_fitness'])
        self.history['avg_fitness'].append(stats['avg_fitness'])
        self.history['best_td_error'].append(stats['best_td_error'])
        self.history['avg_td_error'].append(stats['avg_td_error'])
        self.history['best_return'].append(stats['best_return'])
        self.history['avg_return'].append(stats['avg_return'])

        return stats

    def update_generation(self, new_individuals: List[VEBIndividual]):
        """ô°0 ã"""
        self.individuals = new_individuals
        self.size = len(new_individuals)
        self.generation += 1

    def update_all_target_networks(self):
        """ô°@	*S„îQÜ"""
        for ind in self.individuals:
            ind.update_target_from_q()

    def save(self, path: str):
        """ÝXÍ¤"""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        data = {
            'generation': self.generation,
            'size': self.size,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dims': self.hidden_dims,
            'use_dueling': self.use_dueling,
            'q_weights': np.array([ind.q_weights for ind in self.individuals]),
            'target_weights': np.array([ind.target_weights for ind in self.individuals]),
            'fitnesses': np.array([ind.fitness for ind in self.individuals]),
            'td_errors': np.array([ind.td_error for ind in self.individuals]),
            'returns': np.array([ind.episode_return for ind in self.individuals]),
            'history_best_fitness': np.array(self.history['best_fitness']),
            'history_avg_fitness': np.array(self.history['avg_fitness']),
            'history_best_td_error': np.array(self.history['best_td_error']),
            'history_avg_td_error': np.array(self.history['avg_td_error'])
        }

        np.savez_compressed(path, **data)
        print(f"VEB Population saved to {path}")

    def load(self, path: str):
        """ }Í¤"""
        data = np.load(path, allow_pickle=True)

        self.generation = int(data['generation'])
        self.state_dim = int(data['state_dim'])
        self.action_dim = int(data['action_dim'])
        self.hidden_dims = tuple(data['hidden_dims'])
        self.use_dueling = bool(data['use_dueling'])

        q_weights_array = data['q_weights']
        target_weights_array = data['target_weights']
        fitnesses = data['fitnesses']
        td_errors = data['td_errors']
        returns = data['returns']

        self.individuals = []
        for i in range(len(q_weights_array)):
            ind = VEBIndividual(
                q_weights=q_weights_array[i],
                target_weights=target_weights_array[i],
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dims=self.hidden_dims,
                use_dueling=self.use_dueling
            )
            ind.fitness = fitnesses[i]
            ind.td_error = td_errors[i]
            ind.episode_return = returns[i]
            self.individuals.append(ind)

        self.size = len(self.individuals)

        #  }†ò
        if 'history_best_fitness' in data:
            self.history['best_fitness'] = data['history_best_fitness'].tolist()
            self.history['avg_fitness'] = data['history_avg_fitness'].tolist()
            self.history['best_td_error'] = data['history_best_td_error'].tolist()
            self.history['avg_td_error'] = data['history_avg_td_error'].tolist()

        print(f"VEB Population loaded from {path}, generation {self.generation}")

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, idx):
        return self.individuals[idx]

    def __iter__(self):
        return iter(self.individuals)


# KÕã
if __name__ == "__main__":
    # ú*S
    ind = VEBIndividual()
    print(f"Individual: {ind}")
    print(f"Q weights shape: {ind.q_weights.shape}")
    print(f"Target weights shape: {ind.target_weights.shape}")

    # KÕQÜú
    q_net = ind.create_q_network()
    target_net = ind.create_target_network()

    state = torch.randn(4, 230)
    q_values = q_net(state)
    target_values = target_net(state)
    print(f"Q values shape: {q_values.shape}")
    print(f"Target values shape: {target_values.shape}")

    # KÕÞ>²:
    buffer = ReplayBuffer(capacity=1000)
    for _ in range(100):
        s = np.random.randn(230).astype(np.float32)
        a = np.random.randint(0, 40)
        r = np.random.randn()
        s_next = np.random.randn(230).astype(np.float32)
        done = np.random.random() < 0.1
        buffer.add(s, a, r, s_next, done)

    print(f"\nBuffer size: {len(buffer)}")

    batch = buffer.sample(32)
    print(f"Batch states shape: {batch['states'].shape}")

    # KÕ”¦¡—
    fitness = ind.compute_fitness(buffer, batch_size=32, n_batches=5)
    print(f"\nFitness (negative TD error): {fitness:.4f}")
    print(f"TD error: {ind.td_error:.4f}")

    # KÕÍ¤
    pop = VEBPopulation(size=10)
    pop.initialize_random()
    print(f"\nPopulation size: {len(pop)}")

    # ¡—@	*S”¦
    for individual in pop:
        individual.compute_fitness(buffer, batch_size=32, n_batches=3)

    stats = pop.compute_statistics()
    print(f"Stats: {stats}")

    elite = pop.get_elite(3)
    print(f"Elite fitnesses: {[e.fitness for e in elite]}")

    # KÕÝX/ }
    pop.save("test_veb_pop.npz")
    pop2 = VEBPopulation()
    pop2.load("test_veb_pop.npz")
    print(f"Loaded population size: {len(pop2)}")

    import os
    os.remove("test_veb_pop.npz")
