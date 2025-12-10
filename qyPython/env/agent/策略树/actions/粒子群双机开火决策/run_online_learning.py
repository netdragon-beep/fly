"""
双机协同开火在线学习启动脚本

在真实仿真中边采集数据边训练模型

使用方法:
python run_online_learning.py --episodes 50

特点:
1. ε-greedy 探索策略
2. 优先经验回放
3. 实时模型更新
4. 自适应探索率衰减
"""

import os
import sys
import argparse

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.chdir(project_root)


def main():
    parser = argparse.ArgumentParser(description='双机协同开火在线学习')

    # 运行参数
    parser.add_argument('--episodes', type=int, default=20,
                        help='运行轮数')
    parser.add_argument('--blue_agent', action='store_true',
                        help='是否使用蓝方AI')

    # 在线学习参数
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                        help='初始探索率')
    parser.add_argument('--epsilon_end', type=float, default=0.1,
                        help='最终探索率')
    parser.add_argument('--epsilon_decay', type=float, default=0.995,
                        help='探索率衰减')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='在线更新批大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--update_interval', type=int, default=10,
                        help='每N个新样本更新一次')
    parser.add_argument('--buffer_size', type=int, default=10000,
                        help='经验回放缓冲区大小')

    # 模型参数
    parser.add_argument('--pretrained', type=str, default=None,
                        help='预训练模型路径')
    parser.add_argument('--confidence', type=float, default=0.6,
                        help='开火置信度阈值')

    # GPU
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU设备ID')
    parser.add_argument('--no_cuda', action='store_true',
                        help='禁用CUDA')

    args = parser.parse_args()

    print("=" * 60)
    print("双机协同开火 - 在线学习模式")
    print("=" * 60)

    # 检测GPU
    import torch
    if args.no_cuda:
        device = 'cpu'
    elif torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
        print(f"\nGPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = 'cpu'
    print(f"设备: {device}")

    try:
        import config
        from env.env import Env
        from env.agent.策略树.actions.粒子群双机开火决策.online_learning_agent import (
            OnlineDualFireAgent, OnlineConfig
        )

        # 创建配置
        online_config = OnlineConfig(
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            update_interval=args.update_interval,
            replay_buffer_size=args.buffer_size,
            confidence_threshold=args.confidence,
            device=device
        )

        print(f"\n在线学习配置:")
        print(f"  探索率: {online_config.epsilon_start} -> {online_config.epsilon_end}")
        print(f"  衰减率: {online_config.epsilon_decay}")
        print(f"  批大小: {online_config.batch_size}")
        print(f"  学习率: {online_config.learning_rate}")
        print(f"  更新间隔: {online_config.update_interval}")
        print(f"  缓冲区: {online_config.replay_buffer_size}")
        print(f"  置信度阈值: {online_config.confidence_threshold}")

        # 创建智能体
        red_agent = OnlineDualFireAgent(
            'red', '红方在线学习',
            config=online_config,
            pretrained_model_path=args.pretrained
        )

        # 蓝方
        blue_agent = None
        if args.blue_agent:
            from env.agent.策略树.agent import BTDemoAgent
            blue_agent = BTDemoAgent('blue', '蓝方AI')

        print(f"\n红方: OnlineDualFireAgent (在线学习)")
        print(f"蓝方: {'BTDemoAgent' if blue_agent else 'None'}")

        # 修改运行次数
        original_run_times = config.run_times
        config.run_times = args.episodes

        print(f"\n开始运行 {args.episodes} 轮仿真...")
        print("=" * 60)

        # 运行仿真
        env = Env()
        env.run(red_agent, blue_agent)

        # 恢复配置
        config.run_times = original_run_times

        # 保存结果
        print("\n" + "=" * 60)
        print("训练完成!")
        print("=" * 60)

        # 最终统计
        hit_rate = red_agent.total_hits / red_agent.total_fires if red_agent.total_fires > 0 else 0
        print(f"\n最终统计:")
        print(f"  总样本: {red_agent.total_samples}")
        print(f"  总开火: {red_agent.total_fires}")
        print(f"  命中率: {hit_rate:.1%}")
        print(f"  最终探索率: {red_agent.epsilon:.3f}")

        # 保存模型和数据
        model_path = red_agent.save_model()
        data_path = red_agent.save_data()

        print(f"\n模型: {model_path}")
        print(f"数据: {data_path}")

        # 最终评估
        if red_agent.eval_accuracies:
            print(f"\n模型准确率变化:")
            for i, acc in enumerate(red_agent.eval_accuracies[-5:]):
                print(f"  评估 {len(red_agent.eval_accuracies)-5+i+1}: {acc:.2%}")

    except ImportError as e:
        print(f"\n错误: {e}")
        print("请确保从项目根目录运行")
        return


if __name__ == "__main__":
    main()
