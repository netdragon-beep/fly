"""
QVGA训练/评估启动脚本

使用方法:
    # 离线训练（模拟环境）
    python run_qvga.py --mode train --generations 100

    # 在线评估（与仿真对战）
    python run_qvga.py --mode eval --model ./checkpoints/qvga/best_individual.npz

    # 在线训练（与仿真交互）
    python run_qvga.py --mode train_online
"""
import os
import sys
import argparse
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from env.env import auto_engage_main
from env.agent.demo.demo_auto_agent import DemoAutoAgent
from env.agent.qvga.trainer import QVGATrainer, MockBattleEnvironment
from env.agent.qvga.qvga_agent import QVGAAutoAgent
from env.agent.qvga.individual import Individual
from utilities.yxHttp import YxHttpRequest as yxHttp


def parse_args():
    parser = argparse.ArgumentParser(description='QVGA Training/Evaluation')

    # 模式
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'train_online'],
                        help='Mode: train (offline), eval, train_online')

    # 训练参数
    parser.add_argument('--population-size', type=int, default=50,
                        help='Population size')
    parser.add_argument('--generations', type=int, default=100,
                        help='Number of generations')
    parser.add_argument('--elite-ratio', type=float, default=0.1,
                        help='Elite ratio')
    parser.add_argument('--mutation-prob', type=float, default=0.1,
                        help='Mutation probability')
    parser.add_argument('--mutation-std', type=float, default=0.1,
                        help='Mutation standard deviation')

    # 模型路径
    parser.add_argument('--model', type=str, default=None,
                        help='Model path for evaluation')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint to resume from')
    parser.add_argument('--save-dir', type=str, default='./checkpoints/qvga',
                        help='Save directory')

    # 其他
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args()


def train_offline(args):
    """离线训练（使用模拟环境）"""
    print("=" * 60)
    print("QVGA Offline Training")
    print("=" * 60)

    # 设置随机种子
    np.random.seed(args.seed)

    # 创建训练器
    trainer = QVGATrainer(
        population_size=args.population_size,
        generations=args.generations,
        elite_ratio=args.elite_ratio,
        mutation_prob=args.mutation_prob,
        mutation_std=args.mutation_std,
        device=args.device,
        save_dir=args.save_dir
    )

    # 加载检查点
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    # 创建模拟环境
    mock_env = MockBattleEnvironment()

    # 训练
    best = trainer.train(battle_func=mock_env.battle)

    print(f"\nTraining completed!")
    print(f"Best fitness: {best.fitness:.2f}")
    print(f"Model saved to: {args.save_dir}")


def evaluate(args):
    """评估模式（与仿真对战）"""
    print("=" * 60)
    print("QVGA Evaluation")
    print("=" * 60)

    if args.model is None:
        # 使用默认路径
        args.model = os.path.join(args.save_dir, "best_individual.npz")

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        print("Please train first or specify a valid model path")
        return

    yxHttp.clear_room()

    if config.current_config.battle_mode == config.ENGAGE_MODE_AUTO:
        # 红方使用QVGA智能体
        red_agent = QVGAAutoAgent(
            side='red',
            name='red_qvga',
            model_path=args.model,
            device=args.device
        )
        # 蓝方使用baseline
        blue_agent = DemoAutoAgent('blue', 'blue_demo')

        print(f"Red: QVGAAutoAgent (model: {args.model})")
        print(f"Blue: DemoAutoAgent (baseline)")
        print("=" * 60)

        auto_engage_main(red_agent, blue_agent)
    else:
        print("QVGA evaluation only supports AUTO engage mode")


def train_online(args):
    """在线训练（与仿真交互）"""
    print("=" * 60)
    print("QVGA Online Training")
    print("=" * 60)
    print("Note: Online training is slower but uses real simulation data")
    print("=" * 60)

    # TODO: 实现与仿真环境的交互训练
    # 这需要修改trainer以支持实际的仿真对战

    print("Online training not fully implemented yet.")
    print("Please use offline training first, then evaluate.")


def main():
    args = parse_args()

    if args.mode == 'train':
        train_offline(args)
    elif args.mode == 'eval':
        evaluate(args)
    elif args.mode == 'train_online':
        train_online(args)


if __name__ == "__main__":
    main()
