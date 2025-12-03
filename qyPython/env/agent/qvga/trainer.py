"""
QVGA训练器

主训练循环，整合所有组件
"""
import os
import sys
import time
import numpy as np
import torch
from typing import Dict, List, Optional, Callable
from datetime import datetime

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from .individual import Individual, PolicyNetwork, decode_weights
from .population import Population
from .q_network import QNetwork, SimpleQNetwork, QNetworkTrainer
from .genetic_operators import create_next_generation


class QVGATrainer:
    """
    Q-Value Based Genetic Algorithm 训练器

    核心流程:
    1. 初始化种群
    2. 评估每个个体（对战 + Q值）
    3. 选择、交叉、变异
    4. 更新Q网络
    5. 重复
    """

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        elite_ratio: float = 0.1,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.1,
        mutation_std: float = 0.1,
        q_lr: float = 1e-3,
        gamma: float = 0.99,
        battles_per_eval: int = 3,
        q_weight: float = 0.3,
        battle_weight: float = 0.7,
        device: str = 'auto',
        save_dir: str = './checkpoints/qvga',
        log_freq: int = 1
    ):
        """
        Args:
            population_size: 种群大小
            generations: 进化代数
            elite_ratio: 精英保留比例
            crossover_prob: 交叉概率
            mutation_prob: 变异概率
            mutation_std: 变异标准差
            q_lr: Q网络学习率
            gamma: 折扣因子
            battles_per_eval: 每个个体评估对战次数
            q_weight: Q值在适应度中的权重
            battle_weight: 对战得分在适应度中的权重
            device: 计算设备
            save_dir: 保存目录
            log_freq: 日志频率
        """
        # 超参数
        self.population_size = population_size
        self.generations = generations
        self.elite_ratio = elite_ratio
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.mutation_std = mutation_std
        self.battles_per_eval = battles_per_eval
        self.q_weight = q_weight
        self.battle_weight = battle_weight
        self.log_freq = log_freq

        # 设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")

        # 保存目录
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 策略网络模板
        self.policy_template = PolicyNetwork().to(self.device)

        # Q网络
        self.q_network = SimpleQNetwork().to(self.device)
        self.q_trainer = QNetworkTrainer(
            self.q_network, lr=q_lr, gamma=gamma, device=str(self.device)
        )

        # 种群
        self.population = Population(size=population_size, network_template=self.policy_template)
        self.population.initialize_random()

        # 训练统计
        self.current_generation = 0
        self.best_fitness_ever = float('-inf')
        self.best_individual_ever = None

        # 状态采样缓冲
        self.state_buffer: List[np.ndarray] = []

    def evaluate_individual(self, individual: Individual,
                            battle_func: Optional[Callable] = None) -> Dict[str, float]:
        """
        评估单个个体

        Args:
            individual: 待评估个体
            battle_func: 对战函数，返回(wins, losses, avg_reward, states)

        Returns:
            评估结果字典
        """
        # 1. Q值评估
        if len(self.state_buffer) > 0:
            # 使用缓冲的状态评估
            sample_states = np.array(self.state_buffer[-100:])  # 最近100个状态
            q_values = self.q_trainer.batch_evaluate(sample_states)
            avg_q = np.mean(q_values)
        else:
            avg_q = 0.0

        # 2. 实战评估（如果提供了对战函数）
        wins, losses, avg_reward = 0, 0, 0.0
        if battle_func is not None:
            for _ in range(self.battles_per_eval):
                result = battle_func(individual)
                wins += result.get('win', 0)
                losses += result.get('loss', 0)
                avg_reward += result.get('reward', 0)

                # 收集状态用于Q网络训练
                if 'states' in result:
                    self.state_buffer.extend(result['states'])

                # 添加经验到Q网络
                if 'experiences' in result:
                    for exp in result['experiences']:
                        self.q_trainer.add_experience(**exp)

            avg_reward /= self.battles_per_eval

        # 3. 计算综合适应度
        battle_score = (wins - losses) / max(1, wins + losses)  # [-1, 1]
        battle_score = (battle_score + 1) / 2 * 100  # [0, 100]

        fitness = self.battle_weight * battle_score + self.q_weight * avg_q

        # 更新个体信息
        individual.fitness = fitness
        individual.q_value = avg_q
        individual.battle_score = battle_score
        individual.wins = wins
        individual.losses = losses

        return {
            'fitness': fitness,
            'q_value': avg_q,
            'battle_score': battle_score,
            'wins': wins,
            'losses': losses,
            'avg_reward': avg_reward
        }

    def evaluate_population(self, battle_func: Optional[Callable] = None):
        """评估整个种群"""
        print(f"Evaluating population (generation {self.current_generation})...")

        for i, individual in enumerate(self.population):
            self.evaluate_individual(individual, battle_func)

            if (i + 1) % 10 == 0:
                print(f"  Evaluated {i + 1}/{len(self.population)} individuals")

    def evolve(self):
        """进化一代"""
        # 创建下一代
        new_individuals = create_next_generation(
            self.population.individuals,
            elite_ratio=self.elite_ratio,
            crossover_prob=self.crossover_prob,
            mutation_prob=self.mutation_prob,
            mutation_std=self.mutation_std
        )

        # 更新种群
        self.population.update_generation(new_individuals)
        self.current_generation += 1

    def update_q_network(self, n_updates: int = 10):
        """更新Q网络"""
        total_loss = 0.0
        for _ in range(n_updates):
            loss = self.q_trainer.update()
            total_loss += loss
        return total_loss / n_updates if n_updates > 0 else 0.0

    def train(self, battle_func: Optional[Callable] = None):
        """
        主训练循环

        Args:
            battle_func: 对战函数，签名: (individual) -> dict
                返回: {'win': 0/1, 'loss': 0/1, 'reward': float, 'states': list, 'experiences': list}
        """
        print("=" * 60)
        print("QVGA Training Started")
        print(f"Population: {self.population_size}, Generations: {self.generations}")
        print(f"Device: {self.device}")
        print("=" * 60)

        start_time = time.time()

        for gen in range(self.generations):
            gen_start = time.time()

            # 1. 评估
            self.evaluate_population(battle_func)

            # 2. 统计
            stats = self.population.compute_statistics()

            # 更新最佳个体
            best = self.population.get_best(1)[0]
            if best.fitness > self.best_fitness_ever:
                self.best_fitness_ever = best.fitness
                self.best_individual_ever = best.copy()
                self.save_best()

            # 3. 日志
            if gen % self.log_freq == 0:
                gen_time = time.time() - gen_start
                print(f"\nGeneration {gen + 1}/{self.generations}")
                print(f"  Best Fitness: {stats['best_fitness']:.2f}")
                print(f"  Avg Fitness:  {stats['avg_fitness']:.2f}")
                print(f"  Diversity:    {stats['diversity']:.4f}")
                print(f"  Best Ever:    {self.best_fitness_ever:.2f}")
                print(f"  Time:         {gen_time:.1f}s")

            # 4. Q网络更新
            if len(self.q_trainer.buffer) > 64:
                q_loss = self.update_q_network(n_updates=5)
                if gen % self.log_freq == 0:
                    print(f"  Q Loss:       {q_loss:.4f}")

            # 5. 进化
            if gen < self.generations - 1:
                self.evolve()

            # 6. 定期保存
            if (gen + 1) % 10 == 0:
                self.save_checkpoint(gen + 1)

        # 训练结束
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("Training Completed!")
        print(f"Total Time: {total_time / 60:.1f} minutes")
        print(f"Best Fitness: {self.best_fitness_ever:.2f}")
        print("=" * 60)

        # 保存最终结果
        self.save_checkpoint(self.generations)
        self.save_best()

        return self.best_individual_ever

    def save_checkpoint(self, generation: int):
        """保存检查点"""
        path = os.path.join(self.save_dir, f"checkpoint_gen{generation}.npz")
        self.population.save(path)

        # 保存Q网络
        q_path = os.path.join(self.save_dir, f"q_network_gen{generation}.pt")
        self.q_trainer.save(q_path)

    def save_best(self):
        """保存最佳个体"""
        if self.best_individual_ever is None:
            return

        path = os.path.join(self.save_dir, "best_individual.npz")
        np.savez_compressed(
            path,
            weights=self.best_individual_ever.weights,
            fitness=self.best_individual_ever.fitness,
            generation=self.current_generation
        )
        print(f"Best individual saved to {path}")

    def load_checkpoint(self, path: str):
        """加载检查点"""
        self.population.load(path)
        self.current_generation = self.population.generation

        # 尝试加载Q网络
        q_path = path.replace("checkpoint_", "q_network_").replace(".npz", ".pt")
        if os.path.exists(q_path):
            self.q_trainer.load(q_path)

    def load_best(self, path: str) -> Individual:
        """加载最佳个体"""
        data = np.load(path)
        individual = Individual(weights=data['weights'])
        individual.fitness = float(data['fitness'])
        return individual


class MockBattleEnvironment:
    """
    模拟对战环境（用于测试）

    实际使用时需要替换为真实的仿真环境
    """

    def __init__(self):
        self.policy_network = PolicyNetwork()

    def battle(self, individual: Individual) -> Dict:
        """
        模拟对战

        Args:
            individual: 待评估个体

        Returns:
            对战结果字典
        """
        # 载入权重
        decode_weights(individual.weights, self.policy_network)

        # 模拟若干步
        states = []
        experiences = []
        total_reward = 0.0

        for step in range(50):
            # 生成模拟状态
            state = np.random.randn(230).astype(np.float32)
            states.append(state)

            # 获取动作
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                action = self.policy_network(state_t)

            # 模拟奖励（随机）
            reward = np.random.randn() * 0.1

            # 下一状态
            next_state = np.random.randn(230).astype(np.float32)
            done = step == 49

            total_reward += reward

            experiences.append({
                'state': state,
                'policy_weights': individual.weights,
                'reward': reward,
                'next_state': next_state,
                'done': done
            })

        # 模拟胜负（基于总奖励）
        win = 1 if total_reward > 0 else 0
        loss = 1 - win

        return {
            'win': win,
            'loss': loss,
            'reward': total_reward,
            'states': states,
            'experiences': experiences
        }


# 测试代码
if __name__ == "__main__":
    # 创建训练器
    trainer = QVGATrainer(
        population_size=20,
        generations=5,
        device='cpu'
    )

    # 创建模拟环境
    mock_env = MockBattleEnvironment()

    # 训练
    best = trainer.train(battle_func=mock_env.battle)

    print(f"\nBest individual fitness: {best.fitness:.2f}")
