# -*- coding: utf-8 -*-
"""
VEB-RL /,

: Value-Evolutionary-Based Reinforcement Learning (ICML 2024)

(:
    # (!߯KխA
    python train.py --mock

    # ޥ
    python train.py --real

    # b
    python train.py --real --resume checkpoints/veb/checkpoint_gen50.npz

    # ( Dueling DQN Q
    python train.py --mock --dueling

    # Ip:
    python train.py --mock --population 100 --elite-size 20 --generations 200
"""
import argparse
import os
import sys

# y
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CODE_ROOT = os.path.dirname(PROJECT_ROOT)
DEFAULT_SAVE_DIR = os.path.join(CODE_ROOT, 'checkpoints', 'veb')
sys.path.insert(0, PROJECT_ROOT)

from veb_trainer import VEBTrainer, MockEnvironment
from veb_battle_env import VEBBattleEnvironment
from reward import RewardConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description='VEB-RL Training - Value-Evolutionary-Based Reinforcement Learning'
    )

    # !
    parser.add_argument('--mock', action='store_true',
                        help='Use mock environment for testing')
    parser.add_argument('--real', action='store_true',
                        help='Use real simulation environment')

    # ͤp
    parser.add_argument('--population', type=int, default=50,
                        help='Population size (default: 50)')
    parser.add_argument('--generations', type=int, default=100,
                        help='Number of generations (default: 100)')

    # VEB-RL 8p
    parser.add_argument('--elite-size', type=int, default=10,
                        help='Number of elite individuals N for interaction (default: 10)')
    parser.add_argument('--target-update-freq', type=int, default=10,
                        help='Target network update frequency H in generations (default: 10)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')

    # W p
    parser.add_argument('--elite-ratio', type=float, default=0.1,
                        help='Elite ratio for evolution (default: 0.1)')
    parser.add_argument('--crossover-prob', type=float, default=0.8,
                        help='Crossover probability (default: 0.8)')
    parser.add_argument('--mutation-prob', type=float, default=0.1,
                        help='Mutation probability per gene (default: 0.1)')
    parser.add_argument('--mutation-std', type=float, default=0.1,
                        help='Mutation standard deviation (default: 0.1)')

    # RL p
    parser.add_argument('--rl-lr', type=float, default=1e-3,
                        help='RL Q network learning rate (default: 1e-3)')
    parser.add_argument('--rl-batch-size', type=int, default=256,
                        help='RL batch size (default: 256)')
    parser.add_argument('--rl-updates', type=int, default=100,
                        help='RL updates per generation (default: 100)')
    parser.add_argument('--buffer-capacity', type=int, default=100000,
                        help='Replay buffer capacity (default: 100000)')

    # p
    parser.add_argument('--episodes-per-elite', type=int, default=3,
                        help='Episodes per elite individual (default: 3)')
    parser.add_argument('--max-episode-steps', type=int, default=1000,
                        help='Max steps per episode (default: 1000)')
    parser.add_argument('--epsilon-start', type=float, default=1.0,
                        help='Initial epsilon for exploration (default: 1.0)')
    parser.add_argument('--epsilon-end', type=float, default=0.01,
                        help='Final epsilon (default: 0.01)')
    parser.add_argument('--epsilon-decay', type=float, default=0.995,
                        help='Epsilon decay rate per generation (default: 0.995)')

    # Qp
    parser.add_argument('--state-dim', type=int, default=230,
                        help='State dimension (default: 230)')
    parser.add_argument('--action-dim', type=int, default=60,
                        help='Action dimension (default: 40)')
    parser.add_argument('--hidden-dims', type=str, default='256,256',
                        help='Hidden layer dimensions, comma-separated (default: 256,256)')
    parser.add_argument('--dueling', action='store_true',
                        help='Use Dueling DQN network architecture')

    # Vp
    parser.add_argument('--center-lon', type=float, default=146.1,
                        help='Center zone longitude (default: 146.1)')
    parser.add_argument('--center-lat', type=float, default=33.3,
                        help='Center zone latitude (default: 33.3)')
    parser.add_argument('--center-radius', type=float, default=20000.0,
                        help='Center zone radius in meters (default: 20000)')

    # v
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto/cpu/cuda (default: auto)')
    parser.add_argument('--save-dir', type=str, default=DEFAULT_SAVE_DIR,
                        help=f'Save directory (default: {DEFAULT_SAVE_DIR})')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--log-freq', type=int, default=1,
                        help='Log frequency in generations (default: 1)')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("VEB-RL: Value-Evolutionary-Based Reinforcement Learning")
    print("Based on ICML 2024 Paper")
    print("=" * 60)

    # 㐐B
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(','))

    # MnVp
    reward_config = RewardConfig(
        center_longitude=args.center_lon,
        center_latitude=args.center_lat,
        center_radius=args.center_radius,
    )

    #  VEB-RL h
    trainer = VEBTrainer(
        # ͤp
        population_size=args.population,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        hidden_dims=hidden_dims,
        use_dueling=args.dueling,

        # VEB-RL p
        elite_size=args.elite_size,
        target_update_freq=args.target_update_freq,
        gamma=args.gamma,

        # W p
        elite_ratio=args.elite_ratio,
        crossover_prob=args.crossover_prob,
        mutation_prob=args.mutation_prob,
        mutation_std=args.mutation_std,

        # RL p
        rl_lr=args.rl_lr,
        rl_batch_size=args.rl_batch_size,
        rl_updates_per_gen=args.rl_updates,
        buffer_capacity=args.buffer_capacity,

        # p
        episodes_per_elite=args.episodes_per_elite,
        max_episode_steps=args.max_episode_steps,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,

        # p
        generations=args.generations,
        device=args.device,
        save_dir=args.save_dir,
        log_freq=args.log_freq,
    )

    # b
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # 	鯃
    if args.mock:
        print("\nUsing MOCK environment for testing")
        env = MockEnvironment(state_dim=args.state_dim, action_dim=args.action_dim)
        env_step = env.step
        env_reset = env.reset
    elif args.real:
        print("\nUsing REAL simulation environment")
        env = VEBBattleEnvironment(
            reward_config=reward_config,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            device=args.device,
            max_episode_steps=args.max_episode_steps
        )
        env_step = env.step
        env_reset = env.reset
    else:
        print("\nNo environment specified, using MOCK by default")
        env = MockEnvironment(state_dim=args.state_dim, action_dim=args.action_dim)
        env_step = env.step
        env_reset = env.reset

    # SpMn
    print(f"\nConfiguration:")
    print(f"  Population size: {args.population}")
    print(f"  Elite size (N): {args.elite_size}")
    print(f"  Target update freq (H): {args.target_update_freq}")
    print(f"  Generations: {args.generations}")
    print(f"  Hidden dims: {hidden_dims}")
    print(f"  Dueling: {args.dueling}")
    print(f"  Device: {trainer.device}")
    print(f"  Save dir: {args.save_dir}")

    # ˭
    print("\nStarting training...")
    best_individual = trainer.train(
        env_step_func=env_step,
        env_reset_func=env_reset
    )

    # SpӜ
    print("\n" + "=" * 60)
    print("VEB-RL Training Complete!")
    print("=" * 60)
    print(f"Best fitness (negative TD error): {best_individual.fitness:.4f}")
    print(f"Best TD error: {best_individual.td_error:.4f}")
    print(f"Best episode return: {best_individual.episode_return:.2f}")
    print(f"Total steps: {trainer.total_steps}")
    print(f"Total episodes: {trainer.total_episodes}")
    print(f"Checkpoints saved to: {args.save_dir}")
    print("=" * 60)

    # s
    if hasattr(env, 'close'):
        env.close()


if __name__ == "__main__":
    main()
