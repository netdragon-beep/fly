"""
双机协同开火数据采集器

通过随机策略生成大量双机攻击场景
并记录命中/未命中结果用于训练分类器

两种采集模式:
1. 合成数据: 基于物理模型模拟命中概率
2. 仿真数据: 与真实仿真环境交互获取结果
"""

import os
import json
import math
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

try:
    from .feature_extractor import DualFireFeatureExtractor, DualFireFeatures
except ImportError:
    from feature_extractor import DualFireFeatureExtractor, DualFireFeatures


@dataclass
class DualFireSample:
    """一条双机协同开火样本"""
    features: np.ndarray    # 特征向量
    hit: int                # 命中结果 (0/1)
    hit_count: int          # 命中数量 (0/1/2)
    timestamp: float        # 采集时间戳

    def to_dict(self) -> dict:
        return {
            'features': self.features.tolist(),
            'hit': self.hit,
            'hit_count': self.hit_count,
            'timestamp': self.timestamp
        }

    @staticmethod
    def from_dict(d: dict) -> 'DualFireSample':
        return DualFireSample(
            features=np.array(d['features'], dtype=np.float32),
            hit=d['hit'],
            hit_count=d['hit_count'],
            timestamp=d['timestamp']
        )


class DualFireDataCollector:
    """
    双机协同开火数据采集器

    核心任务: 大量采样 -> 获取命中结果 -> 保存数据
    """

    # 导弹物理参数
    MISSILE_SPEED = 1200  # m/s
    MISSILE_RANGE_MAX = 35000  # m (有人机)
    MISSILE_RANGE_MIN = 25000  # m (无人机)
    MISSILE_NEZ_HEAD = 22000  # 迎头NEZ
    MISSILE_NEZ_TAIL = 12000  # 尾追NEZ

    def __init__(self, data_dir: str = None):
        """
        Args:
            data_dir: 数据保存目录
        """
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.feature_extractor = DualFireFeatureExtractor()
        self.samples: List[DualFireSample] = []

    def generate_synthetic_samples(self, num_samples: int = 10000,
                                   verbose: bool = True) -> List[DualFireSample]:
        """
        生成合成训练样本

        使用物理模型模拟命中概率，不需要真实仿真环境

        Args:
            num_samples: 样本数量
            verbose: 是否打印进度

        Returns:
            采集的样本列表
        """
        if verbose:
            print(f"生成 {num_samples} 条合成双机协同开火样本...")

        samples = []

        for i in range(num_samples):
            # 1. 随机生成场景
            shooter1, shooter2, target = self._generate_random_scenario()

            # 2. 提取特征
            features = self.feature_extractor.extract(shooter1, shooter2, target)

            # 3. 基于物理模型计算命中概率
            hit_prob_1, hit_prob_2 = self._calculate_hit_probability(features)

            # 4. 模拟命中结果
            hit_1 = 1 if random.random() < hit_prob_1 else 0
            hit_2 = 1 if random.random() < hit_prob_2 else 0

            # 双机协同: 至少一发命中就算成功
            hit = 1 if (hit_1 or hit_2) else 0
            hit_count = hit_1 + hit_2

            sample = DualFireSample(
                features=features.to_array(),
                hit=hit,
                hit_count=hit_count,
                timestamp=datetime.now().timestamp()
            )
            samples.append(sample)

            if verbose and (i + 1) % 1000 == 0:
                hit_rate = sum(s.hit for s in samples) / len(samples)
                print(f"  进度: {i+1}/{num_samples}, 命中率: {hit_rate:.2%}")

        self.samples.extend(samples)

        if verbose:
            hit_rate = sum(s.hit for s in samples) / len(samples)
            avg_hit_count = sum(s.hit_count for s in samples) / len(samples)
            print(f"完成! 命中率: {hit_rate:.2%}, 平均命中数: {avg_hit_count:.2f}")

        return samples

    def _generate_random_scenario(self) -> Tuple[Dict, Dict, Dict]:
        """
        随机生成双机攻击场景

        覆盖各种距离、角度、速度组合
        """
        # ===== 目标 =====
        target = {
            'longitude': 146.0 + random.uniform(-0.2, 0.2),
            'latitude': 33.4 + random.uniform(-0.2, 0.2),
            'altitude': random.uniform(2000, 8000),
            'speed': random.uniform(250, 450),
            'heading': random.uniform(-math.pi, math.pi),
            'v_x': 0,  # 后面计算
            'v_y': 0,
            'v_z': random.uniform(-20, 20),
            'roll': random.uniform(-1.4, 1.4),  # 机动强度
            'platform_entity_type': random.choice(['有人机', '无人机', '无人机'])  # 无人机更多
        }
        # 根据航向计算速度分量
        target['v_x'] = target['speed'] * math.sin(target['heading'])
        target['v_y'] = target['speed'] * math.cos(target['heading'])

        # ===== 射手1 =====
        # 相对目标的距离和角度
        dist_1 = random.uniform(5000, 35000)
        bearing_1 = random.uniform(0, 360)  # 从目标看射手1的方位
        alt_diff_1 = random.uniform(-2000, 2000)

        shooter1 = self._create_shooter_at_bearing(
            target, dist_1, bearing_1, alt_diff_1
        )

        # ===== 射手2 =====
        dist_2 = random.uniform(5000, 35000)
        # 角度差: 让分布覆盖各种情况 (0° ~ 180°)
        angle_diff = random.uniform(0, 180)
        bearing_2 = (bearing_1 + angle_diff) % 360
        alt_diff_2 = random.uniform(-2000, 2000)

        shooter2 = self._create_shooter_at_bearing(
            target, dist_2, bearing_2, alt_diff_2
        )

        return shooter1, shooter2, target

    def _create_shooter_at_bearing(self, target: Dict, distance: float,
                                    bearing: float, alt_diff: float) -> Dict:
        """在指定方位创建射手"""
        bearing_rad = math.radians(bearing)

        # 计算经纬度偏移
        h_dist = distance  # 简化: 假设距离主要是水平距离
        lat_offset = (h_dist * math.cos(bearing_rad)) / 111000
        lon_offset = (h_dist * math.sin(bearing_rad)) / (111000 * math.cos(math.radians(target['latitude'])))

        # 射手朝向目标 (大致)
        heading_to_target = math.radians((bearing + 180) % 360)
        # 加一些随机偏差
        heading = heading_to_target + random.uniform(-0.5, 0.5)

        speed = random.uniform(350, 450)

        return {
            'longitude': target['longitude'] + lon_offset,
            'latitude': target['latitude'] + lat_offset,
            'altitude': target['altitude'] + alt_diff,
            'speed': speed,
            'heading': heading,
            'velocity_x': speed * math.sin(heading),
            'velocity_y': speed * math.cos(heading),
            'velocity_z': random.uniform(-10, 10),
            'type': random.choice(['有人机', '无人机'])
        }

    def _calculate_hit_probability(self, features: DualFireFeatures) -> Tuple[float, float]:
        """
        基于物理模型计算命中概率

        考虑因素:
        1. 距离 (越近越准)
        2. 姿态角 (迎头比尾追更准)
        3. 离轴角 (越小越准)
        4. 目标横向速度 (每个射手看到的不同! 关键修正)
        5. 目标机动强度 (越大越难打)
        6. ** 双机角度差 ** (核心! 角度差越大，目标越难躲)
        """
        # ===== 射手1的单独命中概率 =====
        p1_base = self._single_shooter_hit_prob(
            features.dist_1, features.aspect_1,
            features.off_boresight_1, features.closure_rate_1
        )

        # ===== 射手2的单独命中概率 =====
        p2_base = self._single_shooter_hit_prob(
            features.dist_2, features.aspect_2,
            features.off_boresight_2, features.closure_rate_2
        )

        # ===== 目标因素修正 (每个射手分别计算!) =====
        # 射手1看到的目标横向逃逸速度惩罚
        lateral_factor_1 = max(0.3, 1.0 - features.target_v_lateral_1 / 400)
        # 射手2看到的目标横向逃逸速度惩罚
        lateral_factor_2 = max(0.3, 1.0 - features.target_v_lateral_2 / 400)

        # 目标径向速度影响 (正=远离，负=接近)
        # 目标远离射手时命中概率降低
        radial_factor_1 = 1.0 if features.target_v_radial_1 < 0 else max(0.6, 1.0 - features.target_v_radial_1 / 500)
        radial_factor_2 = 1.0 if features.target_v_radial_2 < 0 else max(0.6, 1.0 - features.target_v_radial_2 / 500)

        # 机动强度惩罚 (共同)
        maneuver_factor = max(0.4, 1.0 - features.target_maneuver * 0.5)

        # 有人机更难打 (共同)
        type_factor = 0.85 if features.target_is_manned else 1.0

        # ===== 协同加成 (核心!) =====
        # 双机角度差越大，目标越难同时躲避两发导弹
        # 90°时加成最大
        angle_diff = features.angle_diff
        if angle_diff >= 60:
            # 角度差大于60°时有显著加成
            coop_bonus = 1.0 + 0.3 * min(angle_diff / 90, 1.0)
        elif angle_diff >= 30:
            # 30-60°有小加成
            coop_bonus = 1.0 + 0.1 * (angle_diff - 30) / 30
        else:
            # 角度差太小，几乎没有协同效果
            coop_bonus = 1.0

        # 额外协同加成: 如果两个射手看到的横向速度差异大，说明目标难以同时躲避
        lateral_diff = abs(features.target_v_lateral_1 - features.target_v_lateral_2)
        if lateral_diff > 100:
            # 横向速度差异大，目标难以两边都躲
            lateral_diff_bonus = 1.0 + 0.15 * min(lateral_diff / 200, 1.0)
        else:
            lateral_diff_bonus = 1.0

        # 到达时间差: 时间差太大协同效果减弱
        if features.time_diff > 10:
            time_penalty = max(0.7, 1.0 - (features.time_diff - 10) / 20)
        else:
            time_penalty = 1.0

        # ===== 最终概率 (每个射手分别计算) =====
        common_factor = maneuver_factor * type_factor
        coop_factor = coop_bonus * lateral_diff_bonus * time_penalty

        # 射手1的命中概率
        hit_prob_1 = min(0.95, p1_base * lateral_factor_1 * radial_factor_1 * common_factor * coop_factor)
        # 射手2的命中概率
        hit_prob_2 = min(0.95, p2_base * lateral_factor_2 * radial_factor_2 * common_factor * coop_factor)

        return hit_prob_1, hit_prob_2

    def _single_shooter_hit_prob(self, dist: float, aspect: float,
                                  off_boresight: float, closure_rate: float) -> float:
        """计算单架飞机的基础命中概率"""
        # 距离因素: NEZ内最高，越远越低
        if dist < 8000:
            dist_factor = 0.9
        elif dist < 15000:
            dist_factor = 0.8 - (dist - 8000) / 70000
        elif dist < 25000:
            dist_factor = 0.7 - (dist - 15000) / 50000
        else:
            dist_factor = max(0.1, 0.5 - (dist - 25000) / 50000)

        # 姿态角因素: 迎头(180°)最好，尾追(0°)最差
        aspect_factor = 0.5 + 0.5 * (aspect / 180)

        # 离轴角因素: 0°最好，90°最差
        if off_boresight < 30:
            boresight_factor = 1.0
        elif off_boresight < 60:
            boresight_factor = 0.8
        else:
            boresight_factor = max(0.3, 1.0 - off_boresight / 90)

        # 接近率因素: 正值(接近)好，负值(远离)差
        if closure_rate > 200:
            closure_factor = 1.0
        elif closure_rate > 0:
            closure_factor = 0.8 + 0.2 * closure_rate / 200
        else:
            closure_factor = max(0.5, 0.8 + closure_rate / 400)

        return dist_factor * aspect_factor * boresight_factor * closure_factor

    def add_simulation_sample(self, shooter1: Dict, shooter2: Dict, target: Dict,
                              hit_result: int, hit_count: int = None):
        """
        添加来自真实仿真的样本

        Args:
            shooter1, shooter2, target: 态势数据
            hit_result: 命中结果 (0/1)
            hit_count: 命中数量 (0/1/2), 可选
        """
        features = self.feature_extractor.extract(shooter1, shooter2, target)

        sample = DualFireSample(
            features=features.to_array(),
            hit=hit_result,
            hit_count=hit_count if hit_count is not None else hit_result,
            timestamp=datetime.now().timestamp()
        )
        self.samples.append(sample)

    def save(self, filename: str = None):
        """保存采集的数据"""
        if filename is None:
            filename = f"dual_fire_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = os.path.join(self.data_dir, filename)

        data = {
            'num_samples': len(self.samples),
            'feature_dim': DualFireFeatures.feature_dim(),
            'feature_names': DualFireFeatures.feature_names(),
            'samples': [s.to_dict() for s in self.samples]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"数据已保存: {filepath} ({len(self.samples)} 条样本)")
        return filepath

    def load(self, filename: str):
        """加载数据"""
        filepath = os.path.join(self.data_dir, filename)

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.samples = [DualFireSample.from_dict(s) for s in data['samples']]
        print(f"数据已加载: {filepath} ({len(self.samples)} 条样本)")

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取训练数据 (X, y)"""
        if not self.samples:
            raise ValueError("没有样本数据")

        X = np.array([s.features for s in self.samples])
        y = np.array([s.hit for s in self.samples])

        return X, y

    def get_statistics(self) -> Dict:
        """获取数据统计信息"""
        if not self.samples:
            return {}

        hits = sum(s.hit for s in self.samples)
        total = len(self.samples)

        return {
            'total_samples': total,
            'hit_samples': hits,
            'miss_samples': total - hits,
            'hit_rate': hits / total,
            'avg_hit_count': sum(s.hit_count for s in self.samples) / total
        }


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("双机协同开火数据采集器测试")
    print("=" * 60)

    collector = DualFireDataCollector()

    # 生成合成数据
    samples = collector.generate_synthetic_samples(num_samples=5000)

    # 统计信息
    stats = collector.get_statistics()
    print(f"\n数据统计:")
    print(f"  总样本数: {stats['total_samples']}")
    print(f"  命中率: {stats['hit_rate']:.2%}")
    print(f"  平均命中数: {stats['avg_hit_count']:.2f}")

    # 保存数据
    collector.save("test_dual_fire_data.json")

    # 获取训练数据
    X, y = collector.get_training_data()
    print(f"\n训练数据形状: X={X.shape}, y={y.shape}")
