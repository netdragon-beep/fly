"""
遗传算法算子

包含:
- 选择: 锦标赛选择、轮盘赌、精英保留
- 交叉: 单点交叉、均匀交叉、BLX-α
- 变异: 高斯变异、均匀变异
"""
import numpy as np
from typing import List, Tuple
from .individual import Individual


# ================== 选择算子 ==================

def tournament_selection(population: List[Individual], tournament_size: int = 3) -> Individual:
    """
    锦标赛选择

    随机选择tournament_size个个体，返回最优者
    """
    competitors = np.random.choice(len(population), size=tournament_size, replace=False)
    best_idx = max(competitors, key=lambda i: population[i].fitness)
    return population[best_idx].copy()


def roulette_selection(population: List[Individual]) -> Individual:
    """
    轮盘赌选择

    按适应度比例选择
    """
    fitnesses = np.array([ind.fitness for ind in population])

    # 处理负值: 平移到正数
    min_fit = fitnesses.min()
    if min_fit < 0:
        fitnesses = fitnesses - min_fit + 1e-6

    # 归一化
    probs = fitnesses / fitnesses.sum()

    idx = np.random.choice(len(population), p=probs)
    return population[idx].copy()


def elite_selection(population: List[Individual], elite_ratio: float = 0.1) -> List[Individual]:
    """
    精英保留

    保留适应度最高的elite_ratio比例的个体
    """
    n_elite = max(1, int(len(population) * elite_ratio))
    sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
    return [ind.copy() for ind in sorted_pop[:n_elite]]


def select_parents(population: List[Individual], n_parents: int,
                   method: str = 'tournament', **kwargs) -> List[Individual]:
    """
    选择父代

    Args:
        population: 当前种群
        n_parents: 需要选择的父代数量
        method: 选择方法 ('tournament', 'roulette')

    Returns:
        选中的父代列表
    """
    parents = []

    for _ in range(n_parents):
        if method == 'tournament':
            parent = tournament_selection(population, kwargs.get('tournament_size', 3))
        elif method == 'roulette':
            parent = roulette_selection(population)
        else:
            raise ValueError(f"Unknown selection method: {method}")
        parents.append(parent)

    return parents


# ================== 交叉算子 ==================

def single_point_crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
    """
    单点交叉

    在随机位置切分，交换后半部分
    """
    size = len(parent1.weights)
    point = np.random.randint(1, size)

    child1_weights = np.concatenate([parent1.weights[:point], parent2.weights[point:]])
    child2_weights = np.concatenate([parent2.weights[:point], parent1.weights[point:]])

    child1 = Individual(weights=child1_weights)
    child2 = Individual(weights=child2_weights)

    return child1, child2


def uniform_crossover(parent1: Individual, parent2: Individual,
                      swap_prob: float = 0.5) -> Tuple[Individual, Individual]:
    """
    均匀交叉

    每个基因位置独立决定是否交换
    """
    mask = np.random.random(len(parent1.weights)) < swap_prob

    child1_weights = np.where(mask, parent2.weights, parent1.weights)
    child2_weights = np.where(mask, parent1.weights, parent2.weights)

    child1 = Individual(weights=child1_weights)
    child2 = Individual(weights=child2_weights)

    return child1, child2


def blx_alpha_crossover(parent1: Individual, parent2: Individual,
                        alpha: float = 0.5) -> Tuple[Individual, Individual]:
    """
    BLX-α 交叉

    在父代值的扩展范围内随机采样
    适合连续值优化
    """
    w1, w2 = parent1.weights, parent2.weights

    # 计算范围
    min_val = np.minimum(w1, w2)
    max_val = np.maximum(w1, w2)
    range_val = max_val - min_val

    # 扩展范围
    lower = min_val - alpha * range_val
    upper = max_val + alpha * range_val

    # 随机采样
    child1_weights = np.random.uniform(lower, upper).astype(np.float32)
    child2_weights = np.random.uniform(lower, upper).astype(np.float32)

    child1 = Individual(weights=child1_weights)
    child2 = Individual(weights=child2_weights)

    return child1, child2


def crossover(parent1: Individual, parent2: Individual,
              method: str = 'uniform', **kwargs) -> Tuple[Individual, Individual]:
    """
    交叉操作

    Args:
        parent1, parent2: 父代个体
        method: 交叉方法 ('single_point', 'uniform', 'blx_alpha')

    Returns:
        两个子代个体
    """
    if method == 'single_point':
        return single_point_crossover(parent1, parent2)
    elif method == 'uniform':
        return uniform_crossover(parent1, parent2, kwargs.get('swap_prob', 0.5))
    elif method == 'blx_alpha':
        return blx_alpha_crossover(parent1, parent2, kwargs.get('alpha', 0.5))
    else:
        raise ValueError(f"Unknown crossover method: {method}")


# ================== 变异算子 ==================

def gaussian_mutation(individual: Individual, mutation_rate: float = 0.1,
                      mutation_std: float = 0.1) -> Individual:
    """
    高斯变异

    以mutation_rate的概率对每个基因添加高斯噪声
    """
    mask = np.random.random(len(individual.weights)) < mutation_rate
    noise = np.random.randn(len(individual.weights)) * mutation_std

    new_weights = individual.weights.copy()
    new_weights[mask] += noise[mask]

    return Individual(weights=new_weights)


def uniform_mutation(individual: Individual, mutation_rate: float = 0.1,
                     mutation_range: float = 0.2) -> Individual:
    """
    均匀变异

    以mutation_rate的概率对每个基因添加均匀噪声
    """
    mask = np.random.random(len(individual.weights)) < mutation_rate
    noise = np.random.uniform(-mutation_range, mutation_range, len(individual.weights))

    new_weights = individual.weights.copy()
    new_weights[mask] += noise[mask]

    return Individual(weights=new_weights)


def reset_mutation(individual: Individual, reset_rate: float = 0.01) -> Individual:
    """
    重置变异

    以reset_rate的概率将基因重置为随机值
    适合跳出局部最优
    """
    mask = np.random.random(len(individual.weights)) < reset_rate
    new_weights = individual.weights.copy()
    new_weights[mask] = np.random.randn(mask.sum()) * 0.1

    return Individual(weights=new_weights)


def mutate(individual: Individual, method: str = 'gaussian', **kwargs) -> Individual:
    """
    变异操作

    Args:
        individual: 待变异个体
        method: 变异方法 ('gaussian', 'uniform', 'reset')

    Returns:
        变异后的新个体
    """
    if method == 'gaussian':
        return gaussian_mutation(
            individual,
            kwargs.get('mutation_rate', 0.1),
            kwargs.get('mutation_std', 0.1)
        )
    elif method == 'uniform':
        return uniform_mutation(
            individual,
            kwargs.get('mutation_rate', 0.1),
            kwargs.get('mutation_range', 0.2)
        )
    elif method == 'reset':
        return reset_mutation(individual, kwargs.get('reset_rate', 0.01))
    else:
        raise ValueError(f"Unknown mutation method: {method}")


# ================== 种群操作 ==================

def create_next_generation(population: List[Individual],
                           elite_ratio: float = 0.1,
                           crossover_prob: float = 0.8,
                           mutation_prob: float = 0.1,
                           crossover_method: str = 'uniform',
                           mutation_method: str = 'gaussian',
                           mutation_std: float = 0.1) -> List[Individual]:
    """
    创建下一代种群

    流程:
    1. 精英保留
    2. 选择父代
    3. 交叉产生子代
    4. 变异
    """
    pop_size = len(population)
    new_population = []

    # 1. 精英保留
    elites = elite_selection(population, elite_ratio)
    new_population.extend(elites)

    # 2. 生成剩余个体
    while len(new_population) < pop_size:
        # 选择父代
        parent1 = tournament_selection(population)
        parent2 = tournament_selection(population)

        # 交叉
        if np.random.random() < crossover_prob:
            child1, child2 = crossover(parent1, parent2, method=crossover_method)
        else:
            child1, child2 = parent1.copy(), parent2.copy()

        # 变异
        if np.random.random() < mutation_prob:
            child1 = mutate(child1, method=mutation_method, mutation_std=mutation_std)
        if np.random.random() < mutation_prob:
            child2 = mutate(child2, method=mutation_method, mutation_std=mutation_std)

        new_population.append(child1)
        if len(new_population) < pop_size:
            new_population.append(child2)

    return new_population[:pop_size]


# 测试代码
if __name__ == "__main__":
    # 创建测试种群
    population = [Individual() for _ in range(20)]
    for i, ind in enumerate(population):
        ind.fitness = np.random.randn()  # 随机适应度

    print(f"Initial population size: {len(population)}")
    print(f"Best fitness: {max(ind.fitness for ind in population):.4f}")

    # 测试选择
    parents = select_parents(population, 10, method='tournament')
    print(f"Selected {len(parents)} parents")

    # 测试交叉
    child1, child2 = crossover(parents[0], parents[1], method='uniform')
    print(f"Crossover produced children with shapes: {child1.weights.shape}, {child2.weights.shape}")

    # 测试变异
    mutated = mutate(child1, method='gaussian', mutation_rate=0.1, mutation_std=0.1)
    diff = np.sum(mutated.weights != child1.weights)
    print(f"Mutation changed {diff} genes")

    # 测试下一代生成
    next_gen = create_next_generation(population)
    print(f"Next generation size: {len(next_gen)}")
