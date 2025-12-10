"""
VEB-RL火控策略训练脚本

使用方法:
1. 预训练模式: python train_veb_fire.py --mode pretrain
2. 在线训练模式: python train_veb_fire.py --mode online
"""

import os
import sys
import argparse
import numpy as np
import torch
from datetime import datetime

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from veb_fire_control import VEBFireControl, VEBConfig


def generate_synthetic_data(num_samples: int = 10000):
    """
    生成合成训练数据

    状态维度 (9维):
    - 距离 (km)
    - 方位角 (rad)
    - 高度差 (km)
    - 敌机速度 (km/s)
    - 相对速度 (km/s)
    - 目标航向角 (rad)
    - 导弹剩余数量 (归一化)
    - 我方燃油量 (归一化)
    - 威胁等级 (0-1)
    """
    print(f"生成 {num_samples} 条合成训练数据...")

    experiences = []

    for _ in range(num_samples):
        # 随机生成状态
        distance = np.random.uniform(5, 80)  # 5-80 km
        azimuth = np.random.uniform(-np.pi, np.pi)
        altitude_diff = np.random.uniform(-5, 5)  # -5 to 5 km
        enemy_speed = np.random.uniform(0.2, 0.8)  # 0.2-0.8 km/s
        relative_speed = np.random.uniform(-0.5, 0.5)
        target_heading = np.random.uniform(-np.pi, np.pi)
        missiles_left = np.random.uniform(0, 1)
        fuel = np.random.uniform(0.2, 1.0)
        threat_level = np.random.uniform(0, 1)

        state = np.array([
            distance / 100,  # 归一化
            azimuth / np.pi,
            altitude_diff / 10,
            enemy_speed,
            relative_speed,
            target_heading / np.pi,
            missiles_left,
            fuel,
            threat_level
        ], dtype=np.float32)

        # 基于规则生成动作和奖励
        # NEZ (No Escape Zone) 简化模型
        in_nez = 10 < distance < 40 and abs(azimuth) < np.pi/3
        in_range = 5 < distance < 60

        # 决定是否开火
        if in_nez and missiles_left > 0.2:
            action = 1  # 开火
            # 模拟命中概率
            hit_prob = max(0, 1 - distance/50) * max(0, 1 - abs(azimuth)/(np.pi/2))
            hit = np.random.random() < hit_prob
            reward = 10.0 if hit else -2.0  # 命中大奖励，未命中小惩罚
        elif in_range and missiles_left > 0.5 and np.random.random() < 0.3:
            action = 1  # 有时候在射程内也开火
            reward = -1.0  # 射程边缘开火效果不好
        else:
            action = 0  # 不开火
            reward = 0.1 if not in_nez else -0.5  # 在NEZ内不开火是惩罚

        # 生成下一状态（简化：距离变化）
        next_distance = distance + relative_speed * 1.0  # 1秒后
        next_state = state.copy()
        next_state[0] = next_distance / 100

        # 终止条件
        done = next_distance < 2 or next_distance > 100

        experiences.append((state, action, reward, next_state, done))

    return experiences


def pretrain(config: VEBConfig, num_episodes: int = 100, samples_per_episode: int = 1000):
    """预训练模式"""
    print("=" * 60)
    print("VEB-RL 火控策略预训练")
    print("=" * 60)

    # 创建VEB-RL控制器
    veb = VEBFireControl(config)

    best_fitness = float('-inf')

    for episode in range(num_episodes):
        # 生成训练数据
        experiences = generate_synthetic_data(samples_per_episode)

        # 添加到经验池
        for exp in experiences:
            veb.add_experience(*exp)

        # 训练若干步
        losses = []
        for _ in range(100):  # 每episode训练100步
            loss = veb.update_rl()
            if loss is not None:
                losses.append(loss)

        # 进化种群
        if (episode + 1) % 5 == 0:
            veb.evolve_population()

        # 获取统计信息
        stats = veb.get_training_stats()

        # 打印进度
        avg_loss = np.mean(losses) if losses else 0
        current_fitness = stats['best_fitness']

        print(f"Episode {episode+1}/{num_episodes} | "
              f"Loss: {avg_loss:.4f} | "
              f"Best Fitness: {current_fitness:.4f} | "
              f"Generation: {stats['generation']}")

        # 保存最佳模型
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            save_path = os.path.join(os.path.dirname(__file__), 'models', 'veb_fire_best.pt')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            veb.save(save_path)
            print(f"  -> 保存最佳模型 (fitness: {best_fitness:.4f})")

    # 保存最终模型
    final_path = os.path.join(os.path.dirname(__file__), 'models', 'veb_fire_final.pt')
    veb.save(final_path)
    print(f"\n训练完成! 最终模型保存至: {final_path}")

    return veb


def test_model(model_path: str):
    """测试已训练的模型"""
    print("=" * 60)
    print("VEB-RL 火控策略测试")
    print("=" * 60)

    config = VEBConfig()
    veb = VEBFireControl(config)
    veb.load(model_path)

    # 生成测试场景
    test_scenarios = [
        # (距离, 方位角, 描述)
        (20, 0, "正前方20km - 理想射击位置"),
        (50, 0, "正前方50km - 远距离"),
        (15, np.pi/6, "侧前方15km - 较好位置"),
        (30, np.pi/2, "正侧方30km - 不利位置"),
        (10, 0, "正前方10km - 近距离"),
        (70, 0, "正前方70km - 超远距离"),
    ]

    print("\n测试结果:")
    print("-" * 60)

    for dist, azimuth, desc in test_scenarios:
        state = np.array([
            dist / 100,
            azimuth / np.pi,
            0,  # 高度差
            0.5,  # 敌机速度
            0,  # 相对速度
            0,  # 目标航向
            0.8,  # 导弹剩余
            0.7,  # 燃油
            0.5  # 威胁等级
        ], dtype=np.float32)

        should_fire, confidence, reason = veb.should_fire(state)

        status = "开火!" if should_fire else "等待"
        print(f"{desc}")
        print(f"  -> 决策: {status} | 置信度: {confidence:.2%} | 原因: {reason}")
        print()


def main():
    parser = argparse.ArgumentParser(description='VEB-RL火控策略训练')
    parser.add_argument('--mode', type=str, default='pretrain',
                       choices=['pretrain', 'test'],
                       help='运行模式: pretrain(预训练) 或 test(测试)')
    parser.add_argument('--episodes', type=int, default=100,
                       help='训练episode数量')
    parser.add_argument('--model', type=str, default=None,
                       help='测试时使用的模型路径')
    parser.add_argument('--population', type=int, default=10,
                       help='种群大小')
    parser.add_argument('--elite', type=int, default=3,
                       help='精英数量')

    args = parser.parse_args()

    # 配置
    config = VEBConfig(
        state_dim=9,
        action_dim=2,
        population_size=args.population,
        elite_size=args.elite,
        hidden_dims=(256, 128),
        lr=1e-4,
        gamma=0.99,
        batch_size=64,
        buffer_size=100000,
        evolution_method='ga',
        mutation_rate=0.1,
        mutation_strength=0.05,
        crossover_rate=0.5,
        rl_injection_interval=5,
        target_update_freq=100
    )

    if args.mode == 'pretrain':
        pretrain(config, num_episodes=args.episodes)
    elif args.mode == 'test':
        model_path = args.model
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'veb_fire_best.pt')

        if not os.path.exists(model_path):
            print(f"错误: 模型文件不存在: {model_path}")
            print("请先运行预训练: python train_veb_fire.py --mode pretrain")
            return

        test_model(model_path)


if __name__ == "__main__":
    main()
