#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速训练脚本：用极小配置跑几代，用于验证 VEB-RL 算法逻辑是否正常。
"""

import argparse
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CODE_ROOT = os.path.dirname(os.path.dirname(PROJECT_ROOT))
DEFAULT_SAVE_DIR = os.path.join(CODE_ROOT, 'checkpoints', 'veb', 'quick')

sys.path.insert(0, PROJECT_ROOT)

from reward import RewardConfig  # noqa: E402
from veb_trainer import VEBTrainer, MockEnvironment  # noqa: E402
from veb_battle_env import VEBBattleEnvironment  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description='VEB-RL Quick Training (small config for sanity check)'
    )
    parser.add_argument('--real', action='store_true',
                        help='使用真实仿真环境')
    parser.add_argument('--mock', action='store_true',
                        help='强制使用 mock 环境（默认）')

    parser.add_argument('--population', type=int, default=10,
                        help='种群大小 (default: 10)')
    parser.add_argument('--generations', type=int, default=5,
                        help='训练世代数 (default: 5)')
    parser.add_argument('--elite-size', type=int, default=4,
                        help='参与交互的精英数量 (default: 4)')
    parser.add_argument('--elite-ratio', type=float, default=0.2,
                        help='保留精英比例 (default: 0.2)')
    parser.add_argument('--target-update-freq', type=int, default=2,
                        help='目标网络更新频率 (default: 2)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='折扣因子 (default: 0.99)')
    parser.add_argument('--crossover-prob', type=float, default=0.8,
                        help='交叉概率 (default: 0.8)')
    parser.add_argument('--mutation-prob', type=float, default=0.1,
                        help='变异概率 (default: 0.1)')
    parser.add_argument('--mutation-std', type=float, default=0.1,
                        help='变异标准差 (default: 0.1)')

    parser.add_argument('--episodes-per-elite', type=int, default=1,
                        help='每个精英交互回合数 (default: 1)')
    parser.add_argument('--max-episode-steps', type=int, default=500,
                        help='每回合最大发电步数 (default: 500)')
    parser.add_argument('--epsilon-start', type=float, default=1.0,
                        help='初始探索率 (default: 1.0)')
    parser.add_argument('--epsilon-end', type=float, default=0.05,
                        help='最小探索率 (default: 0.05)')
    parser.add_argument('--epsilon-decay', type=float, default=0.95,
                        help='每代衰减系数 (default: 0.95)')

    parser.add_argument('--rl-lr', type=float, default=1e-3,
                        help='Q 网络学习率 (default: 1e-3)')
    parser.add_argument('--rl-batch-size', type=int, default=128,
                        help='RL 批大小 (default: 128)')
    parser.add_argument('--rl-updates', type=int, default=20,
                        help='每代 RL 更新次数 (default: 20)')
    parser.add_argument('--buffer-capacity', type=int, default=5000,
                        help='Replay Buffer 容量 (default: 5000)')
    parser.add_argument('--fitness-batch-size', type=int, default=128,
                        help='适应度计算批大小 (default: 128)')
    parser.add_argument('--fitness-n-batches', type=int, default=5,
                        help='适应度计算批次数 (default: 5)')

    parser.add_argument('--state-dim', type=int, default=230,
                        help='状态维度 (default: 230)')
    parser.add_argument('--action-dim', type=int, default=60,
                        help='动作维度 (default: 60)')
    parser.add_argument('--hidden-dims', type=str, default='256,256',
                        help='隐藏层配置，逗号分隔 (default: 256,256)')
    parser.add_argument('--dueling', action='store_true',
                        help='启用 Dueling DQN 架构')

    parser.add_argument('--center-lon', type=float, default=146.1)
    parser.add_argument('--center-lat', type=float, default=33.3)
    parser.add_argument('--center-radius', type=float, default=20000.0)

    parser.add_argument('--device', type=str, default='auto',
                        help='auto/cpu/cuda (default: auto)')
    parser.add_argument('--save-dir', type=str, default=DEFAULT_SAVE_DIR,
                        help=f'模型保存目录 (default: {DEFAULT_SAVE_DIR})')
    parser.add_argument('--resume', type=str, default=None,
                        help='从 checkpoint 恢复 (default: None)')
    parser.add_argument('--log-freq', type=int, default=1,
                        help='日志输出间隔 (default: 1)')

    return parser.parse_args()


def resolve_save_dir(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(CODE_ROOT, path))


def build_trainer(args, hidden_dims, save_dir):
    return VEBTrainer(
        population_size=args.population,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        hidden_dims=hidden_dims,
        use_dueling=args.dueling,

        elite_size=args.elite_size,
        target_update_freq=args.target_update_freq,
        gamma=args.gamma,

        elite_ratio=args.elite_ratio,
        crossover_prob=args.crossover_prob,
        mutation_prob=args.mutation_prob,
        mutation_std=args.mutation_std,

        rl_lr=args.rl_lr,
        rl_batch_size=args.rl_batch_size,
        rl_updates_per_gen=args.rl_updates,

        episodes_per_elite=args.episodes_per_elite,
        max_episode_steps=args.max_episode_steps,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,

        generations=args.generations,
        fitness_batch_size=args.fitness_batch_size,
        fitness_n_batches=args.fitness_n_batches,

        buffer_capacity=args.buffer_capacity,
        device=args.device,
        save_dir=save_dir,
        log_freq=args.log_freq
    )


def main():
    args = parse_args()
    hidden_dims = tuple(int(x.strip()) for x in args.hidden_dims.split(',') if x.strip())
    save_dir = resolve_save_dir(args.save_dir)

    reward_config = RewardConfig(
        center_longitude=args.center_lon,
        center_latitude=args.center_lat,
        center_radius=args.center_radius,
    )

    trainer = build_trainer(args, hidden_dims, save_dir)

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    if args.mock or not args.real:
        print("\n[QuickTrain] Using MOCK environment")
        env = MockEnvironment(state_dim=args.state_dim, action_dim=args.action_dim)
        env_step, env_reset = env.step, env.reset
    else:
        print("\n[QuickTrain] Using REAL simulation environment")
        env = VEBBattleEnvironment(
            reward_config=reward_config,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            device=args.device,
            max_episode_steps=args.max_episode_steps
        )
        env_step, env_reset = env.step, env.reset

    print("\n=== Quick Training Configuration ===")
    print(f"Population: {args.population}")
    print(f"Elite size: {args.elite_size}")
    print(f"Generations: {args.generations}")
    print(f"Hidden dims: {hidden_dims}")
    print(f"Dueling: {args.dueling}")
    print(f"Device: {trainer.device}")
    print(f"Save dir: {save_dir}")
    print("====================================\n")

    best = trainer.train(env_step_func=env_step, env_reset_func=env_reset)

    print("\n====================================")
    print("Quick VEB-RL Training Complete!")
    print("====================================")
    print(f"Best fitness: {best.fitness:.4f}")
    print(f"Best TD error: {best.td_error:.4f}")
    print(f"Best episode return: {best.episode_return:.2f}")
    print(f"Total steps: {trainer.total_steps}")
    print(f"Total episodes: {trainer.total_episodes}")
    print(f"Checkpoints saved to: {save_dir}")
    print("====================================")

    if hasattr(env, 'close'):
        env.close()


if __name__ == '__main__':
    main()
