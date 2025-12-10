"""
SAC火控系统训练脚本

功能:
1. 从历史数据预训练
2. 在线与仿真环境交互训练
3. 评估和保存模型

使用方法:
    # 预训练模式 (使用历史数据)
    python train_fire_control.py --pretrain --data kill_data.json --epochs 100

    # 在线训练模式 (与仿真环境交互)
    python train_fire_control.py --online --episodes 1000

    # 评估模式
    python train_fire_control.py --eval --model checkpoints/sac_fire_control.pt
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
import numpy as np

import torch

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from QC1.fly.qyPython.env.agent.策略树.fire_control_rl暂时不打算使用 import SACFireControl, RLFireControlConfig, HybridFireControl


def pretrain_from_history(args):
    """
    从历史数据预训练
    """
    print("=" * 60)
    print("SAC火控系统 - 历史数据预训练")
    print("=" * 60)

    # 配置
    config = RLFireControlConfig(
        hidden_dim=args.hidden_dim,
        actor_lr=args.lr,
        critic_lr=args.lr,
        batch_size=args.batch_size,
        device="cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    print(f"\n配置:")
    print(f"  设备: {config.device}")
    print(f"  隐藏层: {config.hidden_dim}")
    print(f"  学习率: {config.actor_lr}")
    print(f"  批大小: {config.batch_size}")

    # 创建模型
    sac = SACFireControl(config)

    # 加载数据
    data_path = args.data
    if not os.path.isabs(data_path):
        data_path = os.path.join(os.path.dirname(__file__), data_path)

    if not os.path.exists(data_path):
        print(f"\n错误: 数据文件不存在: {data_path}")
        return

    with open(data_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    print(f"\n数据统计:")
    print(f"  总记录数: {len(records)}")
    kills = sum(1 for r in records if r.get('result') == 'kill')
    print(f"  击杀数: {kills} ({kills/len(records)*100:.1f}%)")
    print(f"  脱靶数: {len(records) - kills} ({(len(records)-kills)/len(records)*100:.1f}%)")

    # 预训练
    print(f"\n开始预训练 {args.epochs} 轮...")
    sac.pretrain_from_data(data_path, epochs=args.epochs)

    # 保存模型
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, f"sac_fire_control_pretrained.pt")
    sac.save(model_path)

    # 评估
    print("\n预训练后评估:")
    evaluate_on_data(sac, records)


def evaluate_on_data(sac: SACFireControl, records: list):
    """
    在历史数据上评估模型
    """
    correct = 0
    total = len(records)

    tp, fp, tn, fn = 0, 0, 0, 0

    for record in records:
        # 构造shooter和target
        shooter = {
            'longitude': 146.1,
            'latitude': 33.3,
            'speed': record.get('shooter_speed', 300),
            'heading': 0,
            'altitude': record.get('shooter_alt', 5000),
            'type': record.get('shooter_type', '无人机')
        }

        target = {
            'longitude': 146.1 + record.get('distance', 10000) / 111000,
            'latitude': 33.3,
            'speed': record.get('target_speed', 300),
            'heading': record.get('aspect_angle', 90) * 3.14159 / 180,  # 简化
            'altitude': record.get('target_alt', 5000),
            'platform_entity_type': record.get('target_type', '无人机')
        }

        # 模型预测
        should_fire, conf, reason = sac.should_fire(shooter, target)

        # 实际结果
        actual_kill = record.get('result') == 'kill'

        # 统计
        if should_fire and actual_kill:
            tp += 1  # 预测开火，实际击杀
        elif should_fire and not actual_kill:
            fp += 1  # 预测开火，实际脱靶
        elif not should_fire and actual_kill:
            fn += 1  # 预测不开火，实际能击杀
        else:
            tn += 1  # 预测不开火，实际也会脱靶

    print(f"  混淆矩阵:")
    print(f"    TP(正确开火): {tp}")
    print(f"    FP(错误开火): {fp}")
    print(f"    TN(正确不开火): {tn}")
    print(f"    FN(错过机会): {fn}")

    if tp + fp > 0:
        precision = tp / (tp + fp)
        print(f"  精确率(开火命中率): {precision*100:.1f}%")

    if tp + fn > 0:
        recall = tp / (tp + fn)
        print(f"  召回率(机会把握率): {recall*100:.1f}%")


def online_train(args):
    """
    在线训练模式 - 与仿真环境交互
    """
    print("=" * 60)
    print("SAC火控系统 - 在线训练")
    print("=" * 60)

    # 尝试导入仿真环境
    try:
        from env.env import Env
        from env.agent.demo.demo_auto_agent import DemoAutoAgent
        HAS_SIM = True
    except ImportError:
        HAS_SIM = False
        print("\n警告: 仿真环境不可用，将使用模拟数据训练")

    # 配置
    config = RLFireControlConfig(
        hidden_dim=args.hidden_dim,
        actor_lr=args.lr,
        critic_lr=args.lr,
        batch_size=args.batch_size,
        device="cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    # 创建模型
    sac = SACFireControl(config)

    # 如果有预训练模型，加载
    if args.resume and os.path.exists(args.resume):
        sac.load(args.resume)

    print(f"\n配置:")
    print(f"  设备: {config.device}")
    print(f"  训练轮数: {args.episodes}")
    print(f"  每轮最大步数: {args.max_steps}")

    # 训练统计
    episode_rewards = []
    episode_kills = []

    for episode in range(args.episodes):
        episode_reward = 0
        kills = 0
        steps = 0

        if HAS_SIM:
            # 真实仿真环境训练
            # TODO: 实现与VEB环境的集成
            pass
        else:
            # 模拟训练
            for step in range(args.max_steps):
                # 生成随机状态
                state = np.random.randn(config.state_dim).astype(np.float32) * 0.5 + 0.5
                state = np.clip(state, 0, 1)

                # 模型决策
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(config.device)
                with torch.no_grad():
                    action = sac.actor.get_action(state_tensor, deterministic=False)
                    action = action.cpu().numpy()[0, 0]

                # 模拟奖励 (根据状态特征)
                # 假设: 距离近、迎头、NEZ内更可能击杀
                kill_prob = 0.5
                if state[0] < 0.5:  # 距离近
                    kill_prob += 0.2
                if state[1] < 0.3:  # 迎头
                    kill_prob += 0.1
                if state[3] > 0.5:  # NEZ内
                    kill_prob += 0.2

                # 根据动作和概率计算奖励
                if action > 0.5:  # 决定开火
                    if np.random.random() < kill_prob:
                        reward = 1.0
                        kills += 1
                    else:
                        reward = -1.0
                else:
                    reward = -0.1  # 不开火小惩罚

                # 下一状态
                next_state = np.random.randn(config.state_dim).astype(np.float32) * 0.5 + 0.5
                next_state = np.clip(next_state, 0, 1)

                # 存储经验
                done = step == args.max_steps - 1
                sac.replay_buffer.add(state, np.array([action]), reward, next_state, done)

                # 更新
                if len(sac.replay_buffer) >= config.batch_size:
                    sac.update()

                episode_reward += reward
                steps += 1

        episode_rewards.append(episode_reward)
        episode_kills.append(kills)

        # 打印进度
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_kills = np.mean(episode_kills[-10:])
            print(f"Episode {episode+1}/{args.episodes}: "
                  f"avg_reward={avg_reward:.2f}, avg_kills={avg_kills:.1f}, "
                  f"alpha={sac.alpha.item():.4f}")

        # 保存检查点
        if (episode + 1) % args.save_freq == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            model_path = os.path.join(args.save_dir, f"sac_fire_control_ep{episode+1}.pt")
            sac.save(model_path)

    # 保存最终模型
    os.makedirs(args.save_dir, exist_ok=True)
    final_path = os.path.join(args.save_dir, "sac_fire_control_final.pt")
    sac.save(final_path)

    print("\n训练完成!")
    print(f"最终模型保存至: {final_path}")


def evaluate_model(args):
    """
    评估模式
    """
    print("=" * 60)
    print("SAC火控系统 - 模型评估")
    print("=" * 60)

    # 加载模型
    config = RLFireControlConfig(
        device="cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    sac = SACFireControl(config)

    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        return

    sac.load(args.model)

    # 加载测试数据
    data_path = args.data
    if not os.path.isabs(data_path):
        data_path = os.path.join(os.path.dirname(__file__), data_path)

    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
        print(f"\n在 {len(records)} 条历史数据上评估:")
        evaluate_on_data(sac, records)
    else:
        print(f"警告: 数据文件不存在: {data_path}")

    # 交互式测试
    print("\n交互式测试 (输入q退出):")
    while True:
        try:
            distance = input("  距离(m, 默认10000): ").strip()
            if distance.lower() == 'q':
                break
            distance = float(distance) if distance else 10000

            aspect = input("  姿态角(度, 默认90): ").strip()
            aspect = float(aspect) if aspect else 90

            closure = input("  接近率(m/s, 默认200): ").strip()
            closure = float(closure) if closure else 200

            # 构造输入
            shooter = {
                'longitude': 146.1,
                'latitude': 33.3,
                'speed': 350,
                'heading': 0,
                'altitude': 5000,
                'type': '无人机'
            }

            target = {
                'longitude': 146.1 + distance / 111000,
                'latitude': 33.3,
                'speed': 300,
                'heading': aspect * 3.14159 / 180,
                'altitude': 5000,
                'platform_entity_type': '无人机'
            }

            # 决策
            should_fire, conf, reason = sac.should_fire(shooter, target)
            print(f"  -> 决策: {'开火' if should_fire else '不开火'}, 置信度: {conf:.3f}")
            print()

        except Exception as e:
            print(f"  错误: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description='SAC火控系统训练')

    # 模式选择
    parser.add_argument('--pretrain', action='store_true', help='预训练模式')
    parser.add_argument('--online', action='store_true', help='在线训练模式')
    parser.add_argument('--eval', action='store_true', help='评估模式')

    # 数据和模型
    parser.add_argument('--data', type=str, default='kill_data.json', help='数据文件路径')
    parser.add_argument('--model', type=str, default='', help='模型文件路径(评估用)')
    parser.add_argument('--resume', type=str, default='', help='继续训练的模型路径')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='预训练轮数')
    parser.add_argument('--episodes', type=int, default=1000, help='在线训练轮数')
    parser.add_argument('--max-steps', type=int, default=100, help='每轮最大步数')
    parser.add_argument('--batch-size', type=int, default=256, help='批大小')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--hidden-dim', type=int, default=256, help='隐藏层维度')

    # 保存
    parser.add_argument('--save-dir', type=str, default='checkpoints/sac_fire_control', help='保存目录')
    parser.add_argument('--save-freq', type=int, default=100, help='保存频率')

    # 其他
    parser.add_argument('--cpu', action='store_true', help='强制使用CPU')

    args = parser.parse_args()

    # 执行
    if args.pretrain:
        pretrain_from_history(args)
    elif args.online:
        online_train(args)
    elif args.eval:
        evaluate_model(args)
    else:
        # 默认: 先预训练，再评估
        print("未指定模式，默认执行预训练")
        pretrain_from_history(args)


if __name__ == "__main__":
    main()
