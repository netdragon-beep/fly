"""
VEB-RL 训练启动脚本

基于论文: Value-Evolutionary-Based Reinforcement Learning (ICML 2024)

使用方法:
    # 使用模拟环境测试训练流程
    python train.py --mock

    # 连接真实仿真环境训练
    python train.py --real

    # 从检查点恢复训练
    python train.py --real --resume checkpoints/veb/checkpoint_gen50.npz

    # 使用 Dueling DQN 网络
    python train.py --mock --dueling
"""
import argparse
import os
import sys

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from veb_trainer import VEBTrainer, MockEnvironment
from veb_battle_env import VEBBattleEnvironment
from reward import RewardConfig


def parse_args():
    parser = argparse.ArgumentParser(description='VEB-RL Training')

    # 训练模式
    parser.add_argument('--mock', action='store_true', help='Use mock environment for testing')
    parser.add_argument('--real', action='store_true', help='Use real simulation environment')

    # 种群参数
    parser.add_argument('--population', type=int, default=50, help='Population size')
    parser.add_argument('--generations', type=int, default=100, help='Number of generations')

    # VEB-RL 核心参数
    parser.add_argument('--elite-size', type=int, default=10, help='Number of elite individuals (N)')
    parser.add_argument('--target-update-freq', type=int, default=10, help='Target network update frequency (H)')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')

    # 遗传算法参数
    parser.add_argument('--elite-ratio', type=float, default=0.1, help='Elite ratio for evolution')
    parser.add_argument('--crossover-prob', type=float, default=0.8, help='Crossover probability')
    parser.add_argument('--mutation-prob', type=float, default=0.1, help='Mutation probability')
    parser.add_argument('--mutation-std', type=float, default=0.1, help='Mutation std')

    # RL 参数
    parser.add_argument('--rl-lr', type=float, default=1e-3, help='RL Q network learning rate')
    parser.add_argument('--rl-batch-size', type=int, default=256, help='RL batch size')
    parser.add_argument('--rl-updates', type=int, default=100, help='RL updates per generation')
    parser.add_argument('--buffer-capacity', type=int, default=100000, help='Replay buffer capacity')

    # 交互参数
    parser.add_argument('--episodes-per-elite', type=int, default=3, help='Episodes per elite individual')
    parser.add_argument('--max-episode-steps', type=int, default=1000, help='Max steps per episode')
    parser.add_argument('--epsilon-start', type=float, default=1.0, help='Initial epsilon')
    parser.add_argument('--epsilon-end', type=float, default=0.01, help='Final epsilon')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Epsilon decay rate')

    # 网络参数
    parser.add_argument('--state-dim', type=int, default=230, help='State dimension')
    parser.add_argument('--action-dim', type=int, default=40, help='Action dimension')
    parser.add_argument('--hidden-dims', type=str, default='256,256', help='Hidden layer dimensions')
    parser.add_argument('--dueling', action='store_true', help='Use Dueling DQN network')

    # 奖励参数
    parser.add_argument('--center-lon', type=float, default=146.1, help='Center zone longitude')
    parser.add_argument('--center-lat', type=float, default=33.3, help='Center zone latitude')
    parser.add_argument('--center-radius', type=float, default=20000.0, help='Center zone radius (m)')

    # 其他
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cpu/cuda)')
    parser.add_argument('--save-dir', type=str, default='./checkpoints/veb', help='Save directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--log-freq', type=int, default=1, help='Log frequency')

    return parser.parse_args()


def main():
    args = parse_args()

    # 解析隐藏层维度
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(','))

    # 配置奖励函数
    reward_config = RewardConfig(
        center_longitude=args.center_lon,
        center_latitude=args.center_lat,
        center_radius=args.center_radius,
    )

    # 创建 VEB-RL 训练器
    trainer = VEBTrainer(
        # 种群参数
        population_size=args.population,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        hidden_dims=hidden_dims,
        use_dueling=args.dueling,

        # VEB-RL 参数
        elite_size=args.elite_size,
        target_update_freq=args.target_update_freq,
        gamma=args.gamma,

        # 遗传算法参数
        elite_ratio=args.elite_ratio,
        crossover_prob=args.crossover_prob,
        mutation_prob=args.mutation_prob,
        mutation_std=args.mutation_std,

        # RL 参数
        rl_lr=args.rl_lr,
        rl_batch_size=args.rl_batch_size,
        rl_updates_per_gen=args.rl_updates,
        buffer_capacity=args.buffer_capacity,

        # 交互参数
        episodes_per_elite=args.episodes_per_elite,
        max_episode_steps=args.max_episode_steps,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,

        # 训练参数
        generations=args.generations,
        device=args.device,
        save_dir=args.save_dir,
        log_freq=args.log_freq,
    )

    # 从检查点恢复
    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # 选择环境
    if args.mock:
        print("Using MOCK environment for testing")
        env = MockEnvironment(state_dim=args.state_dim, action_dim=args.action_dim)
        env_step = env.step
        env_reset = env.reset
    elif args.real:
        print("Using REAL simulation environment")
        env = VEBBattleEnvironment(
            reward_config=reward_config,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            device=args.device
        )
        env_step = env.step
        env_reset = env.reset
    else:
        print("No environment specified, using MOCK by default")
        env = MockEnvironment(state_dim=args.state_dim, action_dim=args.action_dim)
        env_step = env.step
        env_reset = env.reset

    # 开始训练
    best_individual = trainer.train(
        env_step_func=env_step,
        env_reset_func=env_reset
    )

    print("\n" + "=" * 60)
    print("VEB-RL Training Complete!")
    print(f"Best fitness (neg TD error): {best_individual.fitness:.4f}")
    print(f"Best TD error: {best_individual.td_error:.4f}")
    print(f"Best episode return: {best_individual.episode_return:.2f}")
    print(f"Checkpoints saved to: {args.save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
