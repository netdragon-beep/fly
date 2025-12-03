"""
QVGA 训练启动脚本

使用方法:
    # 使用模拟环境测试训练流程
    python train.py --mock

    # 连接真实仿真环境训练
    python train.py --real

    # 从检查点恢复训练
    python train.py --real --resume checkpoints/qvga/checkpoint_gen50.npz
"""
import argparse
import os
import sys

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from trainer import QVGATrainer, MockBattleEnvironment
from real_battle_env import RealBattleEnvironment
from reward import RewardConfig


def parse_args():
    parser = argparse.ArgumentParser(description='QVGA Training')

    # 训练模式
    parser.add_argument('--mock', action='store_true', help='Use mock environment for testing')
    parser.add_argument('--real', action='store_true', help='Use real simulation environment')

    # 训练参数
    parser.add_argument('--population', type=int, default=50, help='Population size')
    parser.add_argument('--generations', type=int, default=100, help='Number of generations')
    parser.add_argument('--battles', type=int, default=3, help='Battles per evaluation')

    # 遗传算法参数
    parser.add_argument('--elite-ratio', type=float, default=0.1, help='Elite ratio')
    parser.add_argument('--crossover-prob', type=float, default=0.8, help='Crossover probability')
    parser.add_argument('--mutation-prob', type=float, default=0.1, help='Mutation probability')
    parser.add_argument('--mutation-std', type=float, default=0.1, help='Mutation std')

    # Q网络参数
    parser.add_argument('--q-lr', type=float, default=1e-3, help='Q network learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--q-weight', type=float, default=0.3, help='Q value weight in fitness')

    # 奖励参数
    parser.add_argument('--center-lon', type=float, default=146.1, help='Center zone longitude')
    parser.add_argument('--center-lat', type=float, default=33.3, help='Center zone latitude')
    parser.add_argument('--center-radius', type=float, default=20000.0, help='Center zone radius (m)')

    # 其他
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cpu/cuda)')
    parser.add_argument('--save-dir', type=str, default='./checkpoints/qvga', help='Save directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--log-freq', type=int, default=1, help='Log frequency')

    return parser.parse_args()


def main():
    args = parse_args()

    # 配置奖励函数
    reward_config = RewardConfig(
        center_longitude=args.center_lon,
        center_latitude=args.center_lat,
        center_radius=args.center_radius,
    )

    # 创建训练器
    trainer = QVGATrainer(
        population_size=args.population,
        generations=args.generations,
        elite_ratio=args.elite_ratio,
        crossover_prob=args.crossover_prob,
        mutation_prob=args.mutation_prob,
        mutation_std=args.mutation_std,
        q_lr=args.q_lr,
        gamma=args.gamma,
        battles_per_eval=args.battles,
        q_weight=args.q_weight,
        battle_weight=1.0 - args.q_weight,
        device=args.device,
        save_dir=args.save_dir,
        log_freq=args.log_freq,
    )

    # 从检查点恢复
    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # 选择对战环境
    if args.mock:
        print("Using MOCK environment for testing")
        env = MockBattleEnvironment()
        battle_func = env.battle
    elif args.real:
        print("Using REAL simulation environment")
        env = RealBattleEnvironment(reward_config=reward_config, device=args.device)
        battle_func = env.battle
    else:
        print("No environment specified, using MOCK by default")
        env = MockBattleEnvironment()
        battle_func = env.battle

    # 开始训练
    best_individual = trainer.train(battle_func=battle_func)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best fitness: {best_individual.fitness:.2f}")
    print(f"Checkpoints saved to: {args.save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
