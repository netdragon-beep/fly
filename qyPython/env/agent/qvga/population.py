"""
种群管理

管理个体集合、统计信息、保存/加载
"""
import numpy as np
import os
import json
from typing import List, Optional, Dict, Any
from datetime import datetime

from .individual import Individual, PolicyNetwork


class Population:
    """
    种群类

    管理一组个体，提供统计和操作接口
    """

    def __init__(self, size: int = 50, network_template: PolicyNetwork = None):
        """
        初始化种群

        Args:
            size: 种群大小
            network_template: 策略网络模板
        """
        self.size = size
        self.network_template = network_template or PolicyNetwork()
        self.individuals: List[Individual] = []
        self.generation = 0

        # 统计信息
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'worst_fitness': [],
            'diversity': []
        }

    def initialize_random(self):
        """随机初始化种群"""
        self.individuals = [
            Individual(network=self.network_template)
            for _ in range(self.size)
        ]
        self.generation = 0

    def initialize_from_individuals(self, individuals: List[Individual]):
        """从个体列表初始化"""
        self.individuals = individuals
        self.size = len(individuals)

    def get_best(self, n: int = 1) -> List[Individual]:
        """获取最优的n个个体"""
        sorted_inds = sorted(self.individuals, key=lambda x: x.fitness, reverse=True)
        return sorted_inds[:n]

    def get_worst(self, n: int = 1) -> List[Individual]:
        """获取最差的n个个体"""
        sorted_inds = sorted(self.individuals, key=lambda x: x.fitness)
        return sorted_inds[:n]

    def compute_statistics(self) -> Dict[str, float]:
        """计算种群统计信息"""
        fitnesses = [ind.fitness for ind in self.individuals]

        stats = {
            'best_fitness': max(fitnesses),
            'avg_fitness': np.mean(fitnesses),
            'worst_fitness': min(fitnesses),
            'std_fitness': np.std(fitnesses),
            'diversity': self._compute_diversity()
        }

        # 记录历史
        self.history['best_fitness'].append(stats['best_fitness'])
        self.history['avg_fitness'].append(stats['avg_fitness'])
        self.history['worst_fitness'].append(stats['worst_fitness'])
        self.history['diversity'].append(stats['diversity'])

        return stats

    def _compute_diversity(self) -> float:
        """
        计算种群多样性

        使用权重向量的平均欧氏距离
        """
        if len(self.individuals) < 2:
            return 0.0

        # 采样计算（避免O(n²)复杂度）
        n_samples = min(20, len(self.individuals))
        indices = np.random.choice(len(self.individuals), n_samples, replace=False)

        distances = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                w1 = self.individuals[indices[i]].weights
                w2 = self.individuals[indices[j]].weights
                dist = np.linalg.norm(w1 - w2)
                distances.append(dist)

        return np.mean(distances) if distances else 0.0

    def update_generation(self, new_individuals: List[Individual]):
        """更新到下一代"""
        self.individuals = new_individuals
        self.size = len(new_individuals)
        self.generation += 1

    def save(self, path: str):
        """
        保存种群

        保存所有个体权重和元信息
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        data = {
            'generation': self.generation,
            'size': self.size,
            'timestamp': datetime.now().isoformat(),
            'individuals': []
        }

        for ind in self.individuals:
            ind_data = {
                'weights': ind.weights.tolist(),
                'fitness': ind.fitness,
                'q_value': ind.q_value,
                'battle_score': ind.battle_score,
                'wins': ind.wins,
                'losses': ind.losses
            }
            data['individuals'].append(ind_data)

        # 保存历史
        data['history'] = self.history

        # 保存为numpy格式（更高效）
        np.savez_compressed(
            path,
            generation=self.generation,
            weights=np.array([ind.weights for ind in self.individuals]),
            fitnesses=np.array([ind.fitness for ind in self.individuals]),
            q_values=np.array([ind.q_value for ind in self.individuals]),
            history_best=np.array(self.history['best_fitness']),
            history_avg=np.array(self.history['avg_fitness'])
        )

        print(f"Population saved to {path}")

    def load(self, path: str):
        """加载种群"""
        data = np.load(path, allow_pickle=True)

        self.generation = int(data['generation'])
        weights_array = data['weights']
        fitnesses = data['fitnesses']
        q_values = data['q_values']

        self.individuals = []
        for i in range(len(weights_array)):
            ind = Individual(weights=weights_array[i])
            ind.fitness = fitnesses[i]
            ind.q_value = q_values[i]
            self.individuals.append(ind)

        self.size = len(self.individuals)

        # 加载历史
        if 'history_best' in data:
            self.history['best_fitness'] = data['history_best'].tolist()
            self.history['avg_fitness'] = data['history_avg'].tolist()

        print(f"Population loaded from {path}, generation {self.generation}")

    def save_best(self, path: str, n: int = 1):
        """只保存最优个体"""
        best = self.get_best(n)

        np.savez_compressed(
            path,
            weights=np.array([ind.weights for ind in best]),
            fitnesses=np.array([ind.fitness for ind in best]),
            generation=self.generation
        )

        print(f"Best {n} individuals saved to {path}")

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, idx):
        return self.individuals[idx]

    def __iter__(self):
        return iter(self.individuals)


class PopulationManager:
    """
    种群管理器

    提供高级操作:
    - 多种群管理（岛屿模型）
    - 迁移
    - 自适应参数
    """

    def __init__(self, n_populations: int = 1, pop_size: int = 50):
        self.populations: List[Population] = []
        self.n_populations = n_populations

        for _ in range(n_populations):
            pop = Population(size=pop_size)
            pop.initialize_random()
            self.populations.append(pop)

    def migrate(self, migration_rate: float = 0.1):
        """
        种群间迁移（岛屿模型）

        每个种群向下一个种群发送最优个体
        """
        if self.n_populations < 2:
            return

        n_migrants = max(1, int(self.populations[0].size * migration_rate))

        # 收集每个种群的最优个体
        migrants = []
        for pop in self.populations:
            best = pop.get_best(n_migrants)
            migrants.append([ind.copy() for ind in best])

        # 环形迁移
        for i, pop in enumerate(self.populations):
            source_idx = (i - 1) % self.n_populations
            # 替换最差个体
            worst_indices = np.argsort([ind.fitness for ind in pop.individuals])[:n_migrants]
            for j, idx in enumerate(worst_indices):
                pop.individuals[idx] = migrants[source_idx][j]

    def get_global_best(self) -> Individual:
        """获取所有种群中的最优个体"""
        all_best = []
        for pop in self.populations:
            all_best.extend(pop.get_best(1))
        return max(all_best, key=lambda x: x.fitness)


# 测试代码
if __name__ == "__main__":
    # 创建种群
    pop = Population(size=20)
    pop.initialize_random()
    print(f"Population size: {len(pop)}")

    # 设置随机适应度
    for ind in pop:
        ind.fitness = np.random.randn()

    # 统计
    stats = pop.compute_statistics()
    print(f"Stats: {stats}")

    # 获取最优
    best = pop.get_best(3)
    print(f"Best 3 fitnesses: {[b.fitness for b in best]}")

    # 保存/加载
    pop.save("test_population.npz")
    pop2 = Population()
    pop2.load("test_population.npz")
    print(f"Loaded population size: {len(pop2)}")

    # 清理
    os.remove("test_population.npz")
