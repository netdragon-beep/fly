# -*- coding: utf-8 -*-
"""
VEB-RL h

: Value-Evolutionary-Based Reinforcement Learning (ICML 2024)

8:
1. Elite Interaction: 	*S
2. Negative TD Error Fitness:  = -TD
3. RL Injection: QQeͤ
4. Target Network Update: HQ
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Callable, Optional, Tuple
import os
from datetime import datetime
import copy

# 支持直接运行和作为模块导入
try:
    from .q_network import QNetwork, DuelingQNetwork, encode_q_weights, decode_q_weights
    from .veb_individual import VEBIndividual, VEBPopulation, ReplayBuffer
except ImportError:
    from q_network import QNetwork, DuelingQNetwork, encode_q_weights, decode_q_weights
    from veb_individual import VEBIndividual, VEBPopulation, ReplayBuffer


class GeneticOperators:
    """
    W P

    ( VEB-RL 	\
    """

    @staticmethod
    def tournament_selection(
        population: List[VEBIndividual],
        n_select: int,
        tournament_size: int = 3
    ) -> List[VEBIndividual]:
        """
        &[	

        Args:
            population: ͤ
            n_select: 	p
            tournament_size: &['

        Returns:
            selected: 	-*Sh
        """
        selected = []
        for _ in range(n_select):
            # :	 tournament_size **S
            candidates = np.random.choice(
                len(population), size=tournament_size, replace=False
            )
            # 	؄
            winner_idx = max(candidates, key=lambda i: population[i].fitness)
            selected.append(population[winner_idx].copy())
        return selected

    @staticmethod
    def rank_selection(
        population: List[VEBIndividual],
        n_select: int
    ) -> List[VEBIndividual]:
        """
        
	

        
	
        """
        # 	
        sorted_pop = sorted(population, key=lambda x: x.fitness)
        n = len(sorted_pop)

        # 
        ranks = np.arange(1, n + 1)
        probs = ranks / ranks.sum()

        # 	
        selected = []
        indices = np.random.choice(n, size=n_select, replace=True, p=probs)
        for idx in indices:
            selected.append(sorted_pop[idx].copy())

        return selected

    @staticmethod
    def crossover(
        parent1: VEBIndividual,
        parent2: VEBIndividual,
        crossover_prob: float = 0.8
    ) -> Tuple[VEBIndividual, VEBIndividual]:
        """
        G

        Args:
            parent1, parent2: 6*S
            crossover_prob: ɂ

        Returns:
            child1, child2: P*S
        """
        child1 = parent1.copy()
        child2 = parent2.copy()

        if np.random.random() < crossover_prob:
            # G - :b
            mask = np.random.random(len(parent1.q_weights)) < 0.5

            child1.q_weights = np.where(mask, parent1.q_weights, parent2.q_weights)
            child2.q_weights = np.where(mask, parent2.q_weights, parent1.q_weights)

            # n
            child1.fitness = -float('inf')
            child2.fitness = -float('inf')

        return child1, child2

    @staticmethod
    def sbx_crossover(
        parent1: VEBIndividual,
        parent2: VEBIndividual,
        crossover_prob: float = 0.8,
        eta: float = 20.0
    ) -> Tuple[VEBIndividual, VEBIndividual]:
        """
        !ߌ6 (SBX)

        <
        """
        child1 = parent1.copy()
        child2 = parent2.copy()

        if np.random.random() < crossover_prob:
            for i in range(len(parent1.q_weights)):
                if np.random.random() < 0.5:
                    if abs(parent1.q_weights[i] - parent2.q_weights[i]) > 1e-10:
                        y1 = min(parent1.q_weights[i], parent2.q_weights[i])
                        y2 = max(parent1.q_weights[i], parent2.q_weights[i])

                        rand = np.random.random()
                        beta = 1.0 + (2.0 * (y1) / (y2 - y1 + 1e-10))
                        alpha = 2.0 - beta ** -(eta + 1.0)

                        if rand <= 1.0 / alpha:
                            beta_q = (rand * alpha) ** (1.0 / (eta + 1.0))
                        else:
                            beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))

                        c1 = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
                        c2 = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))

                        child1.q_weights[i] = c1
                        child2.q_weights[i] = c2

            child1.fitness = -float('inf')
            child2.fitness = -float('inf')

        return child1, child2

    @staticmethod
    def mutate(
        individual: VEBIndividual,
        mutation_prob: float = 0.1,
        mutation_std: float = 0.1
    ) -> VEBIndividual:
        """
        د

        Args:
            individual: *S
            mutation_prob: *	
            mutation_std: 

        Returns:
            mutated: *S
        """
        mutated = individual.copy()

        # :	L
        mask = np.random.random(len(mutated.q_weights)) < mutation_prob

        # دj
        noise = np.random.randn(len(mutated.q_weights)) * mutation_std
        mutated.q_weights = mutated.q_weights + mask * noise

        if mask.any():
            mutated.fitness = -float('inf')

        return mutated

    @staticmethod
    def polynomial_mutation(
        individual: VEBIndividual,
        mutation_prob: float = 0.1,
        eta: float = 20.0
    ) -> VEBIndividual:
        """
        y

        <
        """
        mutated = individual.copy()

        for i in range(len(mutated.q_weights)):
            if np.random.random() < mutation_prob:
                y = mutated.q_weights[i]
                rand = np.random.random()

                if rand < 0.5:
                    delta = (2.0 * rand) ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    delta = 1.0 - (2.0 * (1.0 - rand)) ** (1.0 / (eta + 1.0))

                mutated.q_weights[i] = y + delta
                mutated.fitness = -float('inf')

        return mutated


class VEBTrainer:
    """
    VEB-RL h

     Algorithm 1 (VEB-RL) κ
    """

    def __init__(
        self,
        # ͤp
        population_size: int = 50,
        state_dim: int = 230,
        action_dim: int = 60,
        hidden_dims: Tuple[int, ...] = (256, 256),
        use_dueling: bool = False,

        # VEB-RL p
        elite_size: int = 10,  # N: Elite*Sp
        target_update_freq: int = 10,  # H: Qp	
        gamma: float = 0.99,  # cP

        # W p
        elite_ratio: float = 0.1,  # Yԋ
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.1,
        mutation_std: float = 0.1,
        tournament_size: int = 3,

        # RL p
        rl_lr: float = 1e-3,  # QQf`
        rl_batch_size: int = 256,
        rl_updates_per_gen: int = 100,  # RL!p

        # p
        episodes_per_elite: int = 3,  # *񤒄episodep
        max_episode_steps: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,

        # p
        generations: int = 100,
        fitness_batch_size: int = 256,
        fitness_n_batches: int = 10,

        # v
        buffer_capacity: int = 100000,
        device: str = 'auto',
        save_dir: str = './checkpoints/veb',
        log_freq: int = 1
    ):
        # ͤp
        self.population_size = population_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.use_dueling = use_dueling

        # VEB-RL p
        self.elite_size = elite_size
        self.target_update_freq = target_update_freq
        self.gamma = gamma

        # W p
        self.elite_ratio = elite_ratio
        self.n_elite_keep = max(1, int(population_size * elite_ratio))
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.mutation_std = mutation_std
        self.tournament_size = tournament_size

        # RL p
        self.rl_lr = rl_lr
        self.rl_batch_size = rl_batch_size
        self.rl_updates_per_gen = rl_updates_per_gen

        # p
        self.episodes_per_elite = episodes_per_elite
        self.max_episode_steps = max_episode_steps
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # p
        self.generations = generations
        self.fitness_batch_size = fitness_batch_size
        self.fitness_n_batches = fitness_n_batches

        # 
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # XU
        self.save_dir = save_dir
        self.log_freq = log_freq
        os.makedirs(save_dir, exist_ok=True)

        # ͤ
        self.population = VEBPopulation(
            size=population_size,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            use_dueling=use_dueling
        )
        self.population.initialize_random()

        # ό>:
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            state_dim=state_dim
        )

        # ( RL  Q Q
        if use_dueling:
            self.rl_q_net = DuelingQNetwork(
                state_dim, action_dim, hidden_dims
            ).to(self.device)
            self.rl_target_net = DuelingQNetwork(
                state_dim, action_dim, hidden_dims
            ).to(self.device)
        else:
            self.rl_q_net = QNetwork(
                state_dim, action_dim, hidden_dims
            ).to(self.device)
            self.rl_target_net = QNetwork(
                state_dim, action_dim, hidden_dims
            ).to(self.device)

        self.rl_target_net.load_state_dict(self.rl_q_net.state_dict())
        self.rl_optimizer = optim.Adam(self.rl_q_net.parameters(), lr=rl_lr)

        # W P
        self.genetic_ops = GeneticOperators()

        # ߡ
        self.total_steps = 0
        self.total_episodes = 0

    def elite_interaction(
        self,
        env_step_func: Callable,
        env_reset_func: Callable
    ) -> List[float]:
        """
        Elite Interaction

        	؄ N **S

        Args:
            env_step_func: stepp (action) -> (next_state, reward, done, info)
            env_reset_func: resetp () -> state
        """
        # ־*S
        elites = self.population.get_elite(self.elite_size)
        elite_returns: List[float] = []

        for elite in elites:
            #  Q Q
            q_net = elite.create_q_network(self.device)
            q_net.eval()

            total_return = 0.0

            for _ in range(self.episodes_per_elite):
                state = env_reset_func()
                episode_return = 0.0

                for step in range(self.max_episode_steps):
                    # epsilon-greedy \	
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                    if np.random.random() < self.epsilon:
                        action = np.random.randint(0, self.action_dim)
                    else:
                        with torch.no_grad():
                            q_values = q_net(state_tensor)
                            action = q_values.argmax(dim=-1).item()

                    # 
                    next_state, reward, done, info = env_step_func(action)

                    # Xl
                    self.replay_buffer.add(state, action, reward, next_state, done)

                    episode_return += reward
                    self.total_steps += 1

                    if done:
                        break

                    state = next_state

                total_return += episode_return
                self.total_episodes += 1

            # *Sߡ
            elite.episode_return = total_return / max(1, self.episodes_per_elite)
            elite_returns.append(elite.episode_return)

        # p epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )

        return elite_returns

    def evaluate_fitness(self):
        """
        0@	*S

        fitness = -TD_error
        """
        if len(self.replay_buffer) < self.fitness_batch_size:
            print("Warning: Not enough data in replay buffer for fitness evaluation")
            return

        for ind in self.population:
            ind.compute_fitness(
                self.replay_buffer,
                batch_size=self.fitness_batch_size,
                gamma=self.gamma,
                device=self.device,
                n_batches=self.fitness_n_batches
            )

    def rl_optimization(self):
        """
        RL 

        ( DQN  Q Q
        """
        if len(self.replay_buffer) < self.rl_batch_size:
            return 0.0

        total_loss = 0.0

        for _ in range(self.rl_updates_per_gen):
            batch = self.replay_buffer.sample(self.rl_batch_size)

            states = torch.FloatTensor(batch['states']).to(self.device)
            actions = torch.LongTensor(batch['actions']).to(self.device)
            rewards = torch.FloatTensor(batch['rewards']).to(self.device)
            next_states = torch.FloatTensor(batch['next_states']).to(self.device)
            dones = torch.FloatTensor(batch['dones']).to(self.device)

            # Q(s, a)
            q_values = self.rl_q_net(states)
            q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            # max_a' Q_target(s', a')
            with torch.no_grad():
                next_q_values = self.rl_target_net(next_states)
                max_next_q = next_q_values.max(dim=1)[0]
                td_target = rewards + self.gamma * max_next_q * (1 - dones)

            # _1
            loss = nn.functional.mse_loss(q_sa, td_target)

            self.rl_optimizer.zero_grad()
            loss.backward()
            # j
            torch.nn.utils.clip_grad_norm_(self.rl_q_net.parameters(), 1.0)
            self.rl_optimizer.step()

            total_loss += loss.item()

        return total_loss / self.rl_updates_per_gen

    def rl_injection(self):
        """
        RL Injection

         Q Qeͤb*S	
        """
        # *S
        worst = self.population.get_worst(1)[0]
        worst_idx = self.population.individuals.index(worst)

        # *S( Q QC
        new_ind = VEBIndividual(
            q_weights=encode_q_weights(self.rl_q_net),
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=self.hidden_dims,
            use_dueling=self.use_dueling
        )

        # *S
        new_ind.compute_fitness(
            self.replay_buffer,
            batch_size=self.fitness_batch_size,
            gamma=self.gamma,
            device=self.device,
            n_batches=self.fitness_n_batches
        )

        # b*S
        self.population.individuals[worst_idx] = new_ind

    def evolve(self):
        """
        

        	 ->  -> 
        """
        # Y
        elites = self.population.get_elite(self.n_elite_keep)
        new_population = [e.copy() for e in elites]

        # iY*S
        n_offspring = self.population_size - self.n_elite_keep

        # 	6
        parents = self.genetic_ops.tournament_selection(
            self.population.individuals,
            n_offspring,
            self.tournament_size
        )

        # Ɍ
        i = 0
        while len(new_population) < self.population_size:
            if i + 1 < len(parents):
                child1, child2 = self.genetic_ops.crossover(
                    parents[i], parents[i + 1], self.crossover_prob
                )

                child1 = self.genetic_ops.mutate(
                    child1, self.mutation_prob, self.mutation_std
                )
                child2 = self.genetic_ops.mutate(
                    child2, self.mutation_prob, self.mutation_std
                )

                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

                i += 2
            else:
                # U*6
                child = self.genetic_ops.mutate(
                    parents[i], self.mutation_prob, self.mutation_std
                )
                new_population.append(child)
                i += 1

        self.population.update_generation(new_population)

    def update_target_networks(self):
        """
        @	*SQ

         H gL!
        """
        self.population.update_all_target_networks()

        #  RL Q
        self.rl_target_net.load_state_dict(self.rl_q_net.state_dict())

    def train(
        self,
        env_step_func: Callable = None,
        env_reset_func: Callable = None,
        battle_func: Callable = None
    ) -> VEBIndividual:
        """
        VEB-RL ;

        Args:
            env_step_func: stepp
            env_reset_func: resetp
            battle_func: p	(0	

        Returns:
            best: *S
        """
        print("=" * 60)
        print("VEB-RL Training")
        print(f"Population: {self.population_size}")
        print(f"Elite size (N): {self.elite_size}")
        print(f"Target update freq (H): {self.target_update_freq}")
        print(f"Device: {self.device}")
        print("=" * 60)

        # Л battle_func step/reset p
        if battle_func is not None and env_step_func is None:
            env_step_func, env_reset_func = self._wrap_battle_func(battle_func)

        for gen in range(self.generations):
            gen_start = datetime.now()

            # 1. Elite Interaction - *S
            elite_returns: List[float] = []
            if env_step_func is not None and env_reset_func is not None:
                elite_returns = self.elite_interaction(env_step_func, env_reset_func)

            # 2. Evaluate Fitness - 0@	*STD	
            self.evaluate_fitness()

            # 3. RL Optimization -  Q Q
            rl_loss = self.rl_optimization()

            # 4. RL Injection -  Q Qeͤ
            if len(self.replay_buffer) >= self.fitness_batch_size:
                self.rl_injection()

            # 5. Evolution - W 
            self.evolve()

            # 6. Target Network Update -  H Q
            if (gen + 1) % self.target_update_freq == 0:
                self.update_target_networks()
                print(f"  [Target networks updated]")

            # ߡ
            stats = self.population.compute_statistics()
            gen_time = (datetime.now() - gen_start).total_seconds()

            # 
            if (gen + 1) % self.log_freq == 0:
                if elite_returns:
                    best_ret = max(elite_returns)
                    import numpy as _np
                    avg_ret = float(_np.mean(elite_returns))
                else:
                    best_ret = stats['best_return']
                    avg_ret = stats['avg_return']
                print(f"\nGeneration {gen + 1}/{self.generations}")
                print(f"  Best fitness: {stats['best_fitness']:.4f}")
                print(f"  Avg fitness:  {stats['avg_fitness']:.4f}")
                print(f"  Best TD error: {stats['best_td_error']:.4f}")
                print(f"  Best return:  {best_ret:.2f}")
                print(f"  Avg return:   {avg_ret:.2f}")
                print(f"  RL loss: {rl_loss:.4f}")
                print(f"  Buffer size: {len(self.replay_buffer)}")
                print(f"  Epsilon: {self.epsilon:.4f}")
                print(f"  Time: {gen_time:.2f}s")

            # X
            if (gen + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_gen{gen + 1}.npz")

        # XӜ
        self.save_checkpoint("final.npz")
        best = self.population.get_elite(1)[0]
        self.save_best_individual(best, "best_individual.npz")

        return best

    def _wrap_battle_func(self, battle_func: Callable):
        """
        p gym 

        battle_func 8**SӜ
        M step/reset 
        """
        # 9nEe
        # *X

        class BattleEnvWrapper:
            def __init__(wrapper_self, battle_func, state_dim):
                wrapper_self.battle_func = battle_func
                wrapper_self.state_dim = state_dim
                wrapper_self.current_state = None
                wrapper_self.step_count = 0

            def reset(wrapper_self):
                wrapper_self.current_state = np.zeros(wrapper_self.state_dim, dtype=np.float32)
                wrapper_self.step_count = 0
                return wrapper_self.current_state

            def step(wrapper_self, action):
                # :V
                # E(-
                next_state = np.random.randn(wrapper_self.state_dim).astype(np.float32)
                reward = np.random.randn() * 0.1
                wrapper_self.step_count += 1
                done = wrapper_self.step_count >= 100
                info = {}

                wrapper_self.current_state = next_state
                return next_state, reward, done, info

        wrapper = BattleEnvWrapper(battle_func, self.state_dim)
        return wrapper.step, wrapper.reset

    def save_checkpoint(self, filename: str):
        """X"""
        path = os.path.join(self.save_dir, filename)
        self.population.save(path)

        # Xo
        meta_path = os.path.join(self.save_dir, filename.replace('.npz', '_meta.npz'))
        np.savez_compressed(
            meta_path,
            epsilon=self.epsilon,
            total_steps=self.total_steps,
            total_episodes=self.total_episodes,
            rl_q_weights=encode_q_weights(self.rl_q_net)
        )

    def load_checkpoint(self, path: str):
        """}"""
        self.population.load(path)

        meta_path = path.replace('.npz', '_meta.npz')
        if os.path.exists(meta_path):
            meta = np.load(meta_path)
            self.epsilon = float(meta['epsilon'])
            self.total_steps = int(meta['total_steps'])
            self.total_episodes = int(meta['total_episodes'])
            decode_q_weights(meta['rl_q_weights'], self.rl_q_net)
            self.rl_target_net.load_state_dict(self.rl_q_net.state_dict())

    def save_best_individual(self, individual: VEBIndividual, filename: str):
        """X*S"""
        path = os.path.join(self.save_dir, filename)
        np.savez_compressed(
            path,
            q_weights=individual.q_weights,
            target_weights=individual.target_weights,
            fitness=individual.fitness,
            td_error=individual.td_error,
            episode_return=individual.episode_return,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=self.hidden_dims,
            use_dueling=self.use_dueling
        )
        print(f"Best individual saved to {path}")


class MockEnvironment:
    """
    !߯

    (K VEB-RL A
    """

    def __init__(self, state_dim: int = 230, action_dim: int = 60):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state = None
        self.step_count = 0

    def reset(self):
        self.state = np.random.randn(self.state_dim).astype(np.float32)
        self.step_count = 0
        return self.state

    def step(self, action):
        # UlV
        next_state = self.state + np.random.randn(self.state_dim).astype(np.float32) * 0.1
        reward = -np.sum(np.abs(next_state[:10])) * 0.01  # UV

        self.step_count += 1
        done = self.step_count >= 100

        self.state = next_state
        return next_state, reward, done, {}


# K
if __name__ == "__main__":
    print("Testing VEB-RL Trainer...")

    # !߯
    env = MockEnvironment()

    # h
    trainer = VEBTrainer(
        population_size=20,
        elite_size=5,
        generations=5,
        episodes_per_elite=2,
        max_episode_steps=50,
        rl_updates_per_gen=10,
        target_update_freq=2,
        save_dir='./test_veb_checkpoints'
    )

    # 
    best = trainer.train(
        env_step_func=env.step,
        env_reset_func=env.reset
    )

    print(f"\nBest individual: {best}")

    # 
    import shutil
    shutil.rmtree('./test_veb_checkpoints', ignore_errors=True)
