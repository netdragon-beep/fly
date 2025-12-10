"""
双机协同开火决策训练脚本

使用方法:
1. 生成合成数据并训练: python train_dual_fire.py --mode train
2. 测试模型: python train_dual_fire.py --mode test
3. 分析特征重要性: python train_dual_fire.py --mode analyze
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from feature_extractor import DualFireFeatureExtractor, DualFireFeatures
from data_collector import DualFireDataCollector
from classifier import DualFireClassifier


def train(args):
    """训练模式"""
    print("=" * 60)
    print("双机协同开火分类器训练")
    print("=" * 60)

    # 1. 数据采集
    print(f"\n[1/3] 生成训练数据...")
    collector = DualFireDataCollector()
    collector.generate_synthetic_samples(num_samples=args.samples, verbose=True)

    # 保存数据
    data_path = collector.save(f"dual_fire_train_{args.samples}.json")

    # 2. 获取训练数据
    X, y = collector.get_training_data()
    print(f"\n数据统计: {X.shape[0]} 样本, 命中率 {y.mean():.2%}")

    # 3. 训练分类器
    print(f"\n[2/3] 训练 {args.model} 分类器...")
    classifier = DualFireClassifier(model_type=args.model)
    history = classifier.train(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        verbose=True
    )

    # 4. 评估
    print(f"\n[3/3] 评估模型...")
    metrics = classifier.evaluate(X, y)
    print(f"  准确率: {metrics['accuracy']:.4f}")
    print(f"  精确率: {metrics['precision']:.4f}")
    print(f"  召回率: {metrics['recall']:.4f}")
    print(f"  F1分数: {metrics['f1']:.4f}")

    # 5. 保存模型
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    model_ext = '.pt' if args.model == 'mlp' else '.pkl'
    model_path = os.path.join(model_dir, f"dual_fire_{args.model}_best{model_ext}")
    classifier.save(model_path)

    # 6. 绘制训练曲线 (MLP)
    if args.model == 'mlp' and args.plot:
        plot_training_history(history, args)

    print(f"\n训练完成!")
    print(f"  数据: {data_path}")
    print(f"  模型: {model_path}")


def plot_training_history(history, args):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history['loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title('Training Loss')

    # Accuracy
    axes[1].plot(history['accuracy'], label='Train Acc')
    axes[1].plot(history['val_accuracy'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].set_title('Training Accuracy')

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), 'logs', 'training_curve.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    print(f"  训练曲线: {plot_path}")
    plt.close()


def test(args):
    """测试模式"""
    print("=" * 60)
    print("双机协同开火分类器测试")
    print("=" * 60)

    # 加载模型
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    model_ext = '.pt' if args.model == 'mlp' else '.pkl'
    model_path = args.model_path or os.path.join(model_dir, f"dual_fire_{args.model}_best{model_ext}")

    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先训练模型: python train_dual_fire.py --mode train")
        return

    classifier = DualFireClassifier(model_type=args.model)
    classifier.load(model_path)

    # 测试场景
    test_scenarios = [
        # (描述, 射手1位置, 射手2位置, 角度差)
        ("理想协同(角度差90°)", (145.9, 33.5), (146.1, 33.3), 90),
        ("角度差60°", (145.9, 33.45), (146.05, 33.3), 60),
        ("角度差30°", (145.95, 33.48), (146.0, 33.35), 30),
        ("角度差10°(几乎同方向)", (145.95, 33.45), (145.97, 33.43), 10),
        ("远距离协同", (145.7, 33.7), (146.3, 33.1), 90),
        ("近距离协同", (145.98, 33.42), (146.02, 33.38), 90),
    ]

    print("\n测试结果:")
    print("-" * 80)

    for desc, s1_pos, s2_pos, _ in test_scenarios:
        shooter1 = {
            'longitude': s1_pos[0], 'latitude': s1_pos[1], 'altitude': 5000,
            'speed': 400, 'heading': 0.5,
            'velocity_x': 200, 'velocity_y': 346, 'velocity_z': 0
        }
        shooter2 = {
            'longitude': s2_pos[0], 'latitude': s2_pos[1], 'altitude': 5000,
            'speed': 400, 'heading': 2.5,
            'velocity_x': -200, 'velocity_y': 346, 'velocity_z': 0
        }
        target = {
            'longitude': 146.0, 'latitude': 33.4, 'altitude': 5200,
            'speed': 350, 'heading': 1.0,
            'v_x': 100, 'v_y': -300, 'v_z': 0,
            'roll': 0.3, 'platform_entity_type': '无人机'
        }

        should_fire, confidence, reason = classifier.predict(shooter1, shooter2, target)

        print(f"{desc}:")
        print(f"  -> {'开火!' if should_fire else '等待'} (置信度: {confidence:.0%})")
        print(f"  -> {reason}")
        print()


def analyze(args):
    """分析特征重要性"""
    print("=" * 60)
    print("特征重要性分析")
    print("=" * 60)

    # 生成数据
    collector = DualFireDataCollector()
    collector.generate_synthetic_samples(num_samples=10000, verbose=False)
    X, y = collector.get_training_data()

    # 使用XGBoost或随机森林分析特征重要性
    print("\n训练XGBoost分析特征...")
    try:
        classifier = DualFireClassifier(model_type='xgboost')
        classifier.train(X, y, verbose=False)
        importance = classifier.model.feature_importances_
    except ImportError:
        print("XGBoost未安装，使用随机森林...")
        classifier = DualFireClassifier(model_type='random_forest')
        classifier.train(X, y, verbose=False)
        importance = classifier.model.feature_importances_

    # 排序并显示
    feature_names = DualFireFeatures.feature_names()
    sorted_idx = np.argsort(importance)[::-1]

    print("\n特征重要性排名:")
    print("-" * 40)
    for i, idx in enumerate(sorted_idx):
        bar = "█" * int(importance[idx] * 50)
        print(f"{i+1:2}. {feature_names[idx]:20s} {importance[idx]:.4f} {bar}")

    # 分析关键发现
    print("\n关键发现:")
    top_features = [feature_names[i] for i in sorted_idx[:3]]
    print(f"  最重要的3个特征: {', '.join(top_features)}")

    if 'angle_diff' in top_features:
        print("  ✓ 攻击角度差是重要特征，证明双机协同的核心价值!")
    if 'dist_1' in top_features or 'dist_2' in top_features:
        print("  ✓ 距离是重要特征，NEZ范围内命中率更高")


def main():
    parser = argparse.ArgumentParser(description='双机协同开火决策训练')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'analyze'],
                        help='运行模式')
    parser.add_argument('--model', type=str, default='mlp',
                        choices=['mlp', 'xgboost', 'random_forest'],
                        help='分类器类型')
    parser.add_argument('--samples', type=int, default=20000,
                        help='训练样本数量')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数 (MLP)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批大小 (MLP)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率 (MLP)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='测试时使用的模型路径')
    parser.add_argument('--plot', action='store_true',
                        help='是否绘制训练曲线')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'analyze':
        analyze(args)


if __name__ == "__main__":
    main()
