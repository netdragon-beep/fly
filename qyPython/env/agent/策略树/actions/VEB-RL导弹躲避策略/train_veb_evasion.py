"""
VEB-RL导弹躲避策略训练脚本

使用方法:
1. 预训练模式: python train_veb_evasion.py --mode pretrain
2. 测试模式: python train_veb_evasion.py --mode test
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

from veb_evasion import VEBEvasion, VEBEvasionConfig


def generate_synthetic_evasion_data(num_samples: int = 10000):
    """
    生成合成躲避训练数据

    状态维度 (12维):
    - 导弹相对距离 (km, 归一化)
    - 导弹方位角 (rad, 归一化)
    - 导弹俯仰角 (rad, 归一化)
    - 导弹相对速度 (km/s)
    - 导弹接近速度 (km/s)
    - 我方速度 (km/s)
    - 我方高度 (km, 归一化)
    - 我方燃油 (归一化)
    - 威胁等级 (0-1)
    - 导弹剩余飞行时间估计 (s, 归一化)
    - 导弹偏离角 (rad, 归一化)
    - 是否被多枚导弹追踪 (0/1)
    """
    print(f"生成 {num_samples} 条合成躲避训练数据...")

    experiences = []

    for _ in range(num_samples):
        # 随机生成导弹威胁状态
        missile_distance = np.random.uniform(1, 30)  # 1-30 km
        missile_azimuth = np.random.uniform(-np.pi, np.pi)
        missile_pitch = np.random.uniform(-np.pi/4, np.pi/4)
        missile_rel_speed = np.random.uniform(0.5, 2.0)  # 导弹比飞机快
        closing_speed = np.random.uniform(0.3, 1.5)  # 接近速度
        my_speed = np.random.uniform(0.2, 0.6)
        my_altitude = np.random.uniform(2, 15)  # 2-15 km
        fuel = np.random.uniform(0.2, 1.0)
        threat_level = 1.0 - missile_distance / 30  # 距离越近威胁越大
        time_to_impact = missile_distance / max(closing_speed, 0.1)
        deviation_angle = np.random.uniform(0, np.pi/2)  # 导弹偏离角
        multi_threat = np.random.choice([0, 1], p=[0.7, 0.3])

        state = np.array([
            missile_distance / 30,
            missile_azimuth / np.pi,
            missile_pitch / (np.pi/4),
            missile_rel_speed / 2,
            closing_speed / 1.5,
            my_speed / 0.6,
            my_altitude / 15,
            fuel,
            threat_level,
            min(time_to_impact / 30, 1.0),
            deviation_angle / (np.pi/2),
            multi_threat
        ], dtype=np.float32)

        # 基于规则决定最佳躲避动作
        # 动作: 0=保持, 1-4=水平(前后左右), 5-8=垂直组合

        # 简化的躲避逻辑
        if missile_distance < 5:
            # 近距离 - 紧急机动
            if missile_azimuth > 0:
                # 导弹在右边，向左急转
                action = 3  # 左转
            else:
                action = 4  # 右转

            # 如果有高度余量，考虑垂直机动
            if my_altitude > 5 and np.random.random() < 0.5:
                action = 7 if missile_pitch > 0 else 8  # 俯冲或爬升

        elif missile_distance < 15:
            # 中距离 - 侧转规避
            if abs(missile_azimuth) < np.pi/4:
                # 导弹正前方，需要侧转
                action = np.random.choice([3, 4])  # 随机左右转
            else:
                # 继续保持角度
                action = 0
        else:
            # 远距离 - 可以保持或小幅调整
            action = np.random.choice([0, 1, 2], p=[0.6, 0.2, 0.2])

        # 计算奖励
        # 模拟一步后的状态
        action_effects = {
            0: (0, 0),      # 保持
            1: (0.1, 0),    # 前进
            2: (-0.1, 0),   # 后退
            3: (0, -0.3),   # 左转
            4: (0, 0.3),    # 右转
            5: (0.05, -0.15),  # 左前
            6: (0.05, 0.15),   # 右前
            7: (-0.05, -0.15), # 左后(俯冲)
            8: (-0.05, 0.15),  # 右后(爬升)
        }

        dist_change, angle_change = action_effects[action]

        # 导弹继续接近
        next_distance = missile_distance - closing_speed * 1.0 + dist_change * my_speed * 5
        next_azimuth = missile_azimuth + angle_change

        # 计算奖励
        if next_distance < 0.5:
            # 被命中
            reward = -100.0
            done = True
        elif next_distance > missile_distance:
            # 距离增加 - 好的躲避
            reward = 5.0 + (next_distance - missile_distance) * 10
            done = False
        elif abs(next_azimuth) > abs(missile_azimuth):
            # 角度增加 - 可能导致导弹脱靶
            reward = 3.0
            done = False
        else:
            # 距离在减少
            reward = -1.0
            done = False

        # 如果导弹飞过（距离开始增加且角度大）
        if next_distance > missile_distance and abs(next_azimuth) > np.pi/2:
            reward = 50.0  # 成功躲避
            done = True

        # 燃油消耗惩罚
        if action != 0:
            reward -= 0.1

        # 生成下一状态
        next_state = state.copy()
        next_state[0] = max(0, min(1, next_distance / 30))
        next_state[1] = np.clip(next_azimuth / np.pi, -1, 1)
        next_state[8] = 1.0 - next_distance / 30  # 更新威胁等级

        experiences.append((state, action, reward, next_state, done))

    return experiences


def pretrain(config: VEBEvasionConfig, num_episodes: int = 100, samples_per_episode: int = 1000):
    """预训练模式"""
    print("=" * 60)
    print("VEB-RL 导弹躲避策略预训练")
    print("=" * 60)

    # 创建VEB-RL控制器
    veb = VEBEvasion(config)

    best_fitness = float('-inf')

    for episode in range(num_episodes):
        # 生成训练数据
        experiences = generate_synthetic_evasion_data(samples_per_episode)

        # 添加到经验池
        for exp in experiences:
            veb.add_experience(*exp)

        # 训练若干步
        losses = []
        for _ in range(100):
            loss = veb.update_rl()
            if loss is not None:
                losses.append(loss)

        # 进化种群
        if (episode + 1) % 5 == 0:
            veb.evolve_population()

        # 获取统计信息
        stats = veb.get_training_stats()

        avg_loss = np.mean(losses) if losses else 0
        current_fitness = stats['best_fitness']

        print(f"Episode {episode+1}/{num_episodes} | "
              f"Loss: {avg_loss:.4f} | "
              f"Best Fitness: {current_fitness:.4f} | "
              f"Generation: {stats['generation']}")

        # 保存最佳模型
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            save_path = os.path.join(os.path.dirname(__file__), 'models', 'veb_evasion_best.pt')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            veb.save(save_path)
            print(f"  -> 保存最佳模型 (fitness: {best_fitness:.4f})")

    # 保存最终模型
    final_path = os.path.join(os.path.dirname(__file__), 'models', 'veb_evasion_final.pt')
    veb.save(final_path)
    print(f"\n训练完成! 最终模型保存至: {final_path}")

    return veb


def test_model(model_path: str):
    """测试已训练的模型"""
    print("=" * 60)
    print("VEB-RL 导弹躲避策略测试")
    print("=" * 60)

    config = VEBEvasionConfig()
    veb = VEBEvasion(config)
    veb.load(model_path)

    # 动作名称映射
    action_names = {
        0: "保持航向",
        1: "加速前进",
        2: "减速",
        3: "左转规避",
        4: "右转规避",
        5: "左前方机动",
        6: "右前方机动",
        7: "左下俯冲",
        8: "右上爬升"
    }

    # 测试场景
    test_scenarios = [
        # (距离, 方位角, 威胁描述)
        (3, 0, "导弹正前方3km - 紧急!"),
        (10, np.pi/6, "导弹右前方10km"),
        (5, -np.pi/4, "导弹左前方5km"),
        (20, 0, "导弹正前方20km - 远距离"),
        (2, np.pi/2, "导弹正右方2km - 侧向威胁"),
        (8, -np.pi/3, "导弹左前方8km"),
    ]

    print("\n测试结果:")
    print("-" * 60)

    for dist, azimuth, desc in test_scenarios:
        threat_level = 1.0 - dist / 30
        state = np.array([
            dist / 30,
            azimuth / np.pi,
            0,  # pitch
            1.0,  # 导弹相对速度
            0.8,  # 接近速度
            0.5,  # 我方速度
            0.5,  # 高度
            0.7,  # 燃油
            threat_level,
            dist / 20,  # 预计碰撞时间
            0.2,  # 偏离角
            0  # 单一威胁
        ], dtype=np.float32)

        action, q_values = veb.select_action(state)

        print(f"{desc}")
        print(f"  -> 推荐动作: {action_names[action]}")
        print(f"  -> Q值分布: {q_values[:5]}...")  # 显示前5个
        print()


def main():
    parser = argparse.ArgumentParser(description='VEB-RL导弹躲避策略训练')
    parser.add_argument('--mode', type=str, default='pretrain',
                       choices=['pretrain', 'test'],
                       help='运行模式')
    parser.add_argument('--episodes', type=int, default=100,
                       help='训练episode数量')
    parser.add_argument('--model', type=str, default=None,
                       help='测试时使用的模型路径')
    parser.add_argument('--population', type=int, default=10,
                       help='种群大小')

    args = parser.parse_args()

    config = VEBEvasionConfig(
        state_dim=12,
        action_dim=9,
        population_size=args.population,
        elite_size=3,
        hidden_dims=(256, 128),
        lr=1e-4,
        gamma=0.99,
        batch_size=64,
        buffer_size=100000
    )

    if args.mode == 'pretrain':
        pretrain(config, num_episodes=args.episodes)
    elif args.mode == 'test':
        model_path = args.model
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'veb_evasion_best.pt')

        if not os.path.exists(model_path):
            print(f"错误: 模型文件不存在: {model_path}")
            return

        test_model(model_path)


if __name__ == "__main__":
    main()
