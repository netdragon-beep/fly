"""
加速仿真训练脚本

使用多实例并行仿真加速数据采集和训练

使用方法:
1. 配置 config.py 中的 generate_task_num (并行实例数)
2. python train_with_simulation.py --samples 10000
3. 等待数据采集完成，自动训练分类器

核心流程:
1. 启动多个仿真实例
2. 在每个实例中随机执行双机协同开火
3. 收集命中/未命中结果
4. 合并数据训练分类器
"""

import os
import sys
import time
import json
import argparse
import random
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from feature_extractor import DualFireFeatureExtractor, DualFireFeatures
    from data_collector import DualFireDataCollector, DualFireSample
    from classifier import DualFireClassifier
    from sim_data_collector import (
        SimulationDataCollector,
        AcceleratedTrainingController,
        select_fire_targets
    )
except ImportError:
    from .feature_extractor import DualFireFeatureExtractor, DualFireFeatures
    from .data_collector import DualFireDataCollector, DualFireSample
    from .classifier import DualFireClassifier
    from .sim_data_collector import (
        SimulationDataCollector,
        AcceleratedTrainingController,
        select_fire_targets
    )


class DualFireTrainingAgent:
    """
    双机协同开火训练智能体

    嵌入到策略树中，自动采集数据
    """

    def __init__(self, side: str = 'red',
                 fire_probability: float = 0.3,
                 min_fire_interval: float = 5.0):
        """
        Args:
            side: 控制的阵营
            fire_probability: 每帧尝试开火的概率
            min_fire_interval: 最小开火间隔 (仿真秒)
        """
        self.side = side
        self.fire_probability = fire_probability
        self.min_fire_interval = min_fire_interval

        self.collector = SimulationDataCollector()
        self.last_fire_time = 0

        # 随机策略参数
        self.random_move_enabled = True

    def get_cmds(self, side_data: Dict) -> List[Dict]:
        """
        获取控制指令 (策略树接口)

        在采集数据模式下:
        1. 随机移动飞机
        2. 随机尝试双机协同开火
        3. 采集命中结果

        Args:
            side_data: 当前阵营的态势数据

        Returns:
            指令列表
        """
        cmds = []
        sim_time = time.time()  # 使用系统时间作为fallback

        # 更新数据采集器状态
        self.collector.update({'side_list': [side_data]}, sim_time)

        # 随机移动
        if self.random_move_enabled:
            cmds.extend(self._random_move_cmds(side_data))

        # 随机尝试开火
        if sim_time - self.last_fire_time >= self.min_fire_interval:
            if random.random() < self.fire_probability:
                fire_cmd = self._try_dual_fire(side_data, sim_time)
                if fire_cmd:
                    cmds.extend(fire_cmd)
                    self.last_fire_time = sim_time

        return cmds

    def _random_move_cmds(self, side_data: Dict) -> List[Dict]:
        """生成随机移动指令"""
        cmds = []
        for platform in side_data.get('platform_list', []):
            # 随机改变航向
            if random.random() < 0.1:  # 10%概率改变航向
                new_heading = platform.get('heading', 0) + random.uniform(-0.3, 0.3)
                # TODO: 生成航向控制指令
                pass

        return cmds

    def _try_dual_fire(self, side_data: Dict, sim_time: float) -> Optional[List[Dict]]:
        """尝试双机协同开火"""
        observation = {'side_list': [side_data]}

        # 选择开火目标
        result = select_fire_targets(observation, self.side)
        if not result:
            return None

        shooter1_id, shooter2_id, target_id = result

        # 获取平台数据
        shooter1 = None
        shooter2 = None
        target = None

        for p in side_data.get('platform_list', []):
            if p['id'] == shooter1_id:
                shooter1 = p
            elif p['id'] == shooter2_id:
                shooter2 = p

        for t in side_data.get('track_list', []):
            if t['target_id'] == target_id:
                target = t

        if not all([shooter1, shooter2, target]):
            return None

        # 提取特征
        features = self.collector.feature_extractor.extract(shooter1, shooter2, target)

        # 计算预计飞行时间
        avg_dist = (features.dist_1 + features.dist_2) / 2
        expected_tof = avg_dist / 1200

        # 创建待定事件
        from sim_data_collector import PendingFireEvent
        event = PendingFireEvent(
            fire_time=sim_time,
            shooter1_id=shooter1_id,
            shooter2_id=shooter2_id,
            target_id=target_id,
            target_name=target.get('target_name', str(target_id)),
            missile1_name=f"{shooter1['name']}_导弹_{self._get_missile_num(shooter1)}",
            missile2_name=f"{shooter2['name']}_导弹_{self._get_missile_num(shooter2)}",
            features=features.to_array(),
            expected_tof=expected_tof
        )

        self.collector.pending_events.append(event)
        self.collector.total_fires += 1

        # 生成开火指令
        fire_cmds = [
            self._create_attack_cmd(shooter1_id, target_id),
            self._create_attack_cmd(shooter2_id, target_id)
        ]

        print(f"[TrainingAgent] 双机开火: {shooter1['name']} + {shooter2['name']} -> {target['target_name']}")
        print(f"  角度差: {features.angle_diff:.1f}°, 距离: {avg_dist/1000:.1f}km")

        return fire_cmds

    def _get_missile_num(self, platform: Dict) -> int:
        """获取导弹编号"""
        max_ammo = 4 if platform.get('type') == '有人机' else 2
        current_ammo = sum(w.get('quantity', 0) for w in platform.get('weapons', []))
        return max_ammo - current_ammo + 1

    def _create_attack_cmd(self, shooter_id: int, target_id: int) -> Dict:
        """创建攻击指令"""
        # 格式参照 YxScriptTreeFunc.attack_target
        return {
            'json_data': json.dumps({
                'fun': 'attackTarget',
                'platformId': shooter_id,
                'targetId': target_id
            })
        }

    def get_statistics(self) -> Dict:
        """获取采集统计"""
        return self.collector.get_statistics()

    def save_data(self, filename: str = None) -> str:
        """保存采集数据"""
        return self.collector.save(filename)

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取训练数据"""
        return self.collector.get_training_data()


def run_accelerated_training(args):
    """
    运行加速仿真训练

    Args:
        args: 命令行参数
    """
    print("=" * 60)
    print("双机协同开火 - 加速仿真训练 (GPU加速)")
    print("=" * 60)

    # GPU信息
    print(f"\n硬件配置:")
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 else 'cuda')
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  显存: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        print(f"  GPU: 不可用，使用CPU")
    print(f"  设备: {device}")

    # 检查是否可以连接仿真环境
    try:
        import config
        print(f"\n仿真配置:")
        print(f"  并行实例数: {config.generate_task_num}")
        print(f"  目标样本数: {args.samples}")
    except ImportError:
        print("\n警告: 无法导入config，使用默认配置")
        print("  如需连接真实仿真环境，请从项目根目录运行")

    # 创建训练控制器
    controller = AcceleratedTrainingController(
        num_instances=args.instances,
        samples_target=args.samples,
        time_ratio=args.time_ratio
    )

    print(f"\n训练控制器已创建")
    print(f"  目标样本: {args.samples}")
    print(f"  加速倍率: {args.time_ratio}x")

    # 检查是否有检查点可恢复
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'data')
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_')]
    if checkpoints and args.resume:
        latest_checkpoint = sorted(checkpoints)[-1]
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        print(f"\n发现检查点: {latest_checkpoint}")
        controller.load_checkpoint(checkpoint_path)
        print(f"  已加载 {controller.get_total_samples()} 个样本")

    # 如果不是连接真实仿真，使用合成数据演示
    if args.demo:
        print("\n[演示模式] 使用合成数据...")
        run_demo_training(controller, args)
    else:
        print("\n[真实仿真模式] 请确保仿真环境已启动...")
        print("提示: 将 DualFireTrainingAgent 集成到策略树中进行数据采集")
        print("      或使用 --demo 参数运行演示模式")

    # 保存检查点
    if controller.get_total_samples() > 0:
        controller.save_checkpoint()

    # 训练分类器
    if controller.get_total_samples() >= args.min_samples:
        train_classifier_from_data(controller, args)
    else:
        print(f"\n样本不足 ({controller.get_total_samples()}/{args.min_samples})，跳过训练")


def run_demo_training(controller: AcceleratedTrainingController, args):
    """使用合成数据演示训练流程"""
    print("\n生成合成数据进行演示...")

    # 使用合成数据采集器
    from data_collector import DualFireDataCollector
    synthetic_collector = DualFireDataCollector()
    synthetic_collector.generate_synthetic_samples(num_samples=args.samples, verbose=True)

    # 将合成数据转换为控制器格式
    demo_collector = SimulationDataCollector()
    demo_collector.samples = synthetic_collector.samples
    controller.collectors['demo'] = demo_collector

    print(f"\n演示数据生成完成: {len(demo_collector.samples)} 样本")


def train_classifier_from_data(controller: AcceleratedTrainingController, args):
    """从采集的数据训练分类器 (GPU加速)"""
    print("\n" + "=" * 60)
    print("训练分类器 (GPU加速)")
    print("=" * 60)

    # 设置设备
    if torch.cuda.is_available() and args.gpu >= 0:
        device = f'cuda:{args.gpu}'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"\n使用设备: {device}")

    # 合并数据
    X, y = controller.merge_all_data()
    print(f"训练数据: {X.shape[0]} 样本, {X.shape[1]} 特征")
    print(f"正样本率: {y.mean():.1%}")

    # 训练分类器 (指定设备)
    classifier = DualFireClassifier(model_type=args.model, device=device)

    # 开始计时
    start_time = time.time()

    history = classifier.train(
        X, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        verbose=True
    )

    train_time = time.time() - start_time
    print(f"\n训练耗时: {train_time:.1f}秒")

    # 评估
    metrics = classifier.evaluate(X, y)
    print(f"\n训练结果:")
    print(f"  准确率: {metrics['accuracy']:.4f}")
    print(f"  精确率: {metrics['precision']:.4f}")
    print(f"  召回率: {metrics['recall']:.4f}")
    print(f"  F1分数: {metrics['f1']:.4f}")

    # 保存模型
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)

    model_ext = '.pt' if args.model == 'mlp' else '.pkl'
    model_name = f"dual_fire_{args.model}_sim"
    if args.demo:
        model_name += "_demo"
    model_path = os.path.join(model_dir, f"{model_name}{model_ext}")

    classifier.save(model_path)
    print(f"\n模型已保存: {model_path}")


def main():
    parser = argparse.ArgumentParser(description='双机协同开火 - 加速仿真训练 (GPU加速)')

    # 数据采集参数
    parser.add_argument('--samples', type=int, default=10000,
                        help='目标样本数量')
    parser.add_argument('--min_samples', type=int, default=1000,
                        help='训练所需最小样本数')
    parser.add_argument('--instances', type=int, default=4,
                        help='并行仿真实例数')
    parser.add_argument('--time_ratio', type=int, default=10,
                        help='仿真加速倍率')

    # 训练参数
    parser.add_argument('--model', type=str, default='mlp',
                        choices=['mlp', 'xgboost', 'random_forest'],
                        help='分类器类型')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批大小 (GPU建议128或256)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')

    # GPU参数
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU设备ID (默认0，-1表示自动选择)')
    parser.add_argument('--no_cuda', action='store_true',
                        help='禁用CUDA，强制使用CPU')

    # 控制参数
    parser.add_argument('--demo', action='store_true',
                        help='使用合成数据演示')
    parser.add_argument('--resume', action='store_true',
                        help='从检查点恢复')

    args = parser.parse_args()

    # 处理CUDA禁用
    if args.no_cuda:
        args.gpu = -999  # 特殊值表示禁用
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    run_accelerated_training(args)


if __name__ == "__main__":
    main()
