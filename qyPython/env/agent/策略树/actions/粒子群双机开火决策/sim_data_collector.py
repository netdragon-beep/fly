"""
真实仿真环境数据采集器

与仿真环境对接，采集真实的双机协同开火数据
支持加速仿真训练

核心流程:
1. 在仿真中随机选择两架我方飞机和一个敌方目标
2. 执行双机协同开火
3. 等待命中/未命中结果
4. 记录数据用于分类器训练
"""

import os
import json
import time
import random
import math
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
from collections import deque
import threading

try:
    from .feature_extractor import DualFireFeatureExtractor, DualFireFeatures
    from .data_collector import DualFireSample
except ImportError:
    from feature_extractor import DualFireFeatureExtractor, DualFireFeatures
    from data_collector import DualFireSample


@dataclass
class PendingFireEvent:
    """待定的开火事件 - 等待命中结果"""
    fire_time: float              # 开火时刻的仿真时间
    shooter1_id: int              # 射手1 ID
    shooter2_id: int              # 射手2 ID
    target_id: int                # 目标 ID
    target_name: str              # 目标名称
    missile1_name: str            # 导弹1名称 (射手1发射)
    missile2_name: str            # 导弹2名称 (射手2发射)
    features: np.ndarray          # 开火时刻的特征
    expected_tof: float           # 预计飞行时间 (秒)
    resolved: bool = False        # 是否已确定结果
    hit_1: int = 0                # 导弹1命中结果
    hit_2: int = 0                # 导弹2命中结果


class SimulationDataCollector:
    """
    真实仿真数据采集器

    核心功能:
    1. 在仿真环境中触发双机协同开火
    2. 跟踪导弹飞行并判断命中结果
    3. 保存真实数据用于训练

    使用方法:
    1. 将此采集器集成到策略树中
    2. 在合适时机调用 try_fire_and_collect()
    3. 每帧调用 update() 更新状态
    """

    # 导弹参数
    MISSILE_SPEED = 1200  # m/s
    MISSILE_MAX_FLIGHT_TIME = 40  # 秒
    MISSILE_HIT_RADIUS = 30  # 命中判定半径 (米)

    def __init__(self, data_dir: str = None, auto_save_interval: int = 100):
        """
        Args:
            data_dir: 数据保存目录
            auto_save_interval: 每采集N个样本自动保存一次
        """
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.feature_extractor = DualFireFeatureExtractor()
        self.samples: List[DualFireSample] = []
        self.pending_events: List[PendingFireEvent] = []

        self.auto_save_interval = auto_save_interval
        self._samples_since_save = 0

        # 统计
        self.total_fires = 0
        self.total_hits = 0
        self.session_start_time = datetime.now()

        # 用于跟踪导弹状态
        self._missile_status: Dict[str, str] = {}  # missile_name -> 'flying'/'hit'/'miss'

        # 线程锁 (多实例并行时使用)
        self._lock = threading.Lock()

    def try_fire_and_collect(self, observation: Dict, fun_tool,
                              shooter1_id: int, shooter2_id: int,
                              target_id: int) -> bool:
        """
        尝试双机协同开火并采集数据

        Args:
            observation: 当前态势数据 (side_list格式)
            fun_tool: YxScriptTreeFunc 实例，用于发送开火指令
            shooter1_id: 射手1的平台ID
            shooter2_id: 射手2的平台ID
            target_id: 目标的ID

        Returns:
            是否成功发起开火
        """
        # 获取shooter和target数据
        shooter1 = self._find_platform_by_id(observation, shooter1_id)
        shooter2 = self._find_platform_by_id(observation, shooter2_id)
        target = self._find_track_by_id(observation, target_id)

        if not all([shooter1, shooter2, target]):
            return False

        # 检查弹药
        if not self._has_ammo(shooter1) or not self._has_ammo(shooter2):
            return False

        # 提取特征
        features = self.feature_extractor.extract(shooter1, shooter2, target)

        # 计算预计飞行时间
        avg_dist = (features.dist_1 + features.dist_2) / 2
        expected_tof = avg_dist / self.MISSILE_SPEED

        # 生成导弹名称 (根据仿真系统命名规则)
        missile1_name = f"{shooter1['name']}_导弹_{self._get_next_missile_num(shooter1)}"
        missile2_name = f"{shooter2['name']}_导弹_{self._get_next_missile_num(shooter2)}"

        # 发送开火指令
        sim_time = observation.get('header', {}).get('sim_time', 0)
        if hasattr(observation, 'get') and 'header' not in observation:
            # 可能是side数据，需要从外层获取sim_time
            sim_time = time.time()  # 使用当前时间作为fallback

        # 创建待定事件
        event = PendingFireEvent(
            fire_time=sim_time,
            shooter1_id=shooter1_id,
            shooter2_id=shooter2_id,
            target_id=target_id,
            target_name=target.get('target_name', str(target_id)),
            missile1_name=missile1_name,
            missile2_name=missile2_name,
            features=features.to_array(),
            expected_tof=expected_tof
        )

        # 执行开火
        try:
            # 调用仿真系统的开火接口
            fun_tool.attack_target(shooter1_id, target_id)
            fun_tool.attack_target(shooter2_id, target_id)

            with self._lock:
                self.pending_events.append(event)
                self.total_fires += 1

            return True

        except Exception as e:
            print(f"[SimCollector] 开火失败: {e}")
            return False

    def update(self, observation: Dict, sim_time: float = None):
        """
        每帧更新 - 检查待定事件的结果

        Args:
            observation: 当前态势数据
            sim_time: 当前仿真时间
        """
        if not self.pending_events:
            return

        if sim_time is None:
            sim_time = observation.get('header', {}).get('sim_time', time.time())

        # 获取当前的broken_list (已损毁单位)
        broken_list = self._get_broken_list(observation)

        # 获取当前的track_list (在飞的导弹)
        track_list = self._get_track_list(observation)
        flying_missiles = {t['target_name'] for t in track_list
                         if t.get('platform_entity_type') == '导弹'}

        with self._lock:
            resolved_events = []

            for event in self.pending_events:
                if event.resolved:
                    continue

                # 检查目标是否被击毁
                target_destroyed = event.target_name in broken_list

                # 检查导弹状态
                missile1_flying = event.missile1_name in flying_missiles
                missile2_flying = event.missile2_name in flying_missiles
                missile1_broken = event.missile1_name in broken_list
                missile2_broken = event.missile2_name in broken_list

                # 判断命中
                # 如果目标被击毁，且导弹消失，认为命中
                if target_destroyed:
                    # 目标被毁，至少一发命中
                    if not missile1_flying and not missile1_broken:
                        event.hit_1 = 1
                    if not missile2_flying and not missile2_broken:
                        event.hit_2 = 1

                # 检查是否可以确定结果
                time_elapsed = sim_time - event.fire_time
                max_wait_time = event.expected_tof + 10  # 额外等待10秒

                # 结果确定条件:
                # 1. 目标已被毁
                # 2. 两发导弹都已消失 (命中或脱靶)
                # 3. 超时
                missiles_resolved = (not missile1_flying and not missile2_flying)
                timeout = time_elapsed > max_wait_time

                if target_destroyed or missiles_resolved or timeout:
                    event.resolved = True
                    resolved_events.append(event)

            # 处理已确定结果的事件
            for event in resolved_events:
                self._record_sample(event)
                self.pending_events.remove(event)

    def _record_sample(self, event: PendingFireEvent):
        """记录一条样本"""
        hit = 1 if (event.hit_1 or event.hit_2) else 0
        hit_count = event.hit_1 + event.hit_2

        sample = DualFireSample(
            features=event.features,
            hit=hit,
            hit_count=hit_count,
            timestamp=datetime.now().timestamp()
        )

        self.samples.append(sample)
        self._samples_since_save += 1

        if hit:
            self.total_hits += 1

        # 打印进度
        if len(self.samples) % 10 == 0:
            hit_rate = self.total_hits / self.total_fires if self.total_fires > 0 else 0
            print(f"[SimCollector] 样本: {len(self.samples)}, "
                  f"命中率: {hit_rate:.1%}, 待定: {len(self.pending_events)}")

        # 自动保存
        if self._samples_since_save >= self.auto_save_interval:
            self.save()
            self._samples_since_save = 0

    def _find_platform_by_id(self, observation: Dict, platform_id: int) -> Optional[Dict]:
        """根据ID查找平台"""
        # 处理不同的observation格式
        if 'platform_list' in observation:
            # 直接是side数据
            for p in observation.get('platform_list', []):
                if p.get('id') == platform_id:
                    return p
        elif 'side_list' in observation:
            # 完整的observation
            for side in observation.get('side_list', []):
                for p in side.get('platform_list', []):
                    if p.get('id') == platform_id:
                        return p
        return None

    def _find_track_by_id(self, observation: Dict, target_id: int) -> Optional[Dict]:
        """根据ID查找探测目标"""
        if 'track_list' in observation:
            for t in observation.get('track_list', []):
                if t.get('target_id') == target_id:
                    return t
        elif 'side_list' in observation:
            for side in observation.get('side_list', []):
                for t in side.get('track_list', []):
                    if t.get('target_id') == target_id:
                        return t
        return None

    def _get_broken_list(self, observation: Dict) -> set:
        """获取已损毁单位列表"""
        broken = set()
        if 'broken_list' in observation:
            broken.update(observation.get('broken_list', []))
        elif 'side_list' in observation:
            for side in observation.get('side_list', []):
                broken.update(side.get('broken_list', []))
        return broken

    def _get_track_list(self, observation: Dict) -> List[Dict]:
        """获取探测目标列表"""
        if 'track_list' in observation:
            return observation.get('track_list', [])
        elif 'side_list' in observation:
            tracks = []
            for side in observation.get('side_list', []):
                tracks.extend(side.get('track_list', []))
            return tracks
        return []

    def _has_ammo(self, platform: Dict) -> bool:
        """检查平台是否有弹药"""
        for weapon in platform.get('weapons', []):
            if weapon.get('quantity', 0) > 0:
                return True
        return False

    def _get_next_missile_num(self, platform: Dict) -> int:
        """获取下一发导弹的编号"""
        # 根据剩余弹药推算已发射数量
        max_ammo = 4 if platform.get('type') == '有人机' else 2
        current_ammo = 0
        for weapon in platform.get('weapons', []):
            current_ammo += weapon.get('quantity', 0)
        return max_ammo - current_ammo + 1

    def save(self, filename: str = None):
        """保存数据"""
        if not self.samples:
            return None

        if filename is None:
            filename = f"sim_dual_fire_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = os.path.join(self.data_dir, filename)

        data = {
            'source': 'simulation',
            'num_samples': len(self.samples),
            'feature_dim': DualFireFeatures.feature_dim(),
            'feature_names': DualFireFeatures.feature_names(),
            'session_start': self.session_start_time.isoformat(),
            'total_fires': self.total_fires,
            'total_hits': self.total_hits,
            'samples': [s.to_dict() for s in self.samples]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[SimCollector] 数据已保存: {filepath} ({len(self.samples)} 样本)")
        return filepath

    def load(self, filename: str):
        """加载数据"""
        filepath = os.path.join(self.data_dir, filename)

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.samples = [DualFireSample.from_dict(s) for s in data['samples']]
        print(f"[SimCollector] 数据已加载: {filepath} ({len(self.samples)} 样本)")

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取训练数据"""
        if not self.samples:
            raise ValueError("没有样本数据")

        X = np.array([s.features for s in self.samples])
        y = np.array([s.hit for s in self.samples])
        return X, y

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if not self.samples:
            return {}

        hits = sum(s.hit for s in self.samples)
        total = len(self.samples)

        return {
            'total_samples': total,
            'hit_samples': hits,
            'miss_samples': total - hits,
            'hit_rate': hits / total if total > 0 else 0,
            'avg_hit_count': sum(s.hit_count for s in self.samples) / total if total > 0 else 0,
            'pending_events': len(self.pending_events),
            'session_duration': (datetime.now() - self.session_start_time).total_seconds()
        }


class AcceleratedTrainingController:
    """
    加速仿真训练控制器

    核心功能:
    1. 并行运行多个仿真实例
    2. 随机生成开火场景
    3. 自动采集数据
    4. 支持断点续训

    使用方法:
    与 multi_env.py 配合使用
    """

    def __init__(self, num_instances: int = 4,
                 samples_target: int = 10000,
                 time_ratio: int = 10):
        """
        Args:
            num_instances: 并行仿真实例数
            samples_target: 目标样本数量
            time_ratio: 仿真加速倍率
        """
        self.num_instances = num_instances
        self.samples_target = samples_target
        self.time_ratio = time_ratio

        # 每个实例的数据采集器
        self.collectors: Dict[str, SimulationDataCollector] = {}

        # 合并的样本
        self.merged_samples: List[DualFireSample] = []

        # 训练状态
        self.is_training = False
        self.start_time = None

    def create_collector(self, room_id: str) -> SimulationDataCollector:
        """为仿真实例创建数据采集器"""
        collector = SimulationDataCollector()
        self.collectors[room_id] = collector
        return collector

    def get_collector(self, room_id: str) -> Optional[SimulationDataCollector]:
        """获取实例的数据采集器"""
        return self.collectors.get(room_id)

    def merge_all_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """合并所有实例的数据"""
        all_samples = []
        for collector in self.collectors.values():
            all_samples.extend(collector.samples)

        if not all_samples:
            raise ValueError("没有采集到数据")

        X = np.array([s.features for s in all_samples])
        y = np.array([s.hit for s in all_samples])

        print(f"[AccelController] 合并数据: {len(all_samples)} 样本, "
              f"命中率: {y.mean():.1%}")

        return X, y

    def get_total_samples(self) -> int:
        """获取总样本数"""
        return sum(len(c.samples) for c in self.collectors.values())

    def get_progress(self) -> float:
        """获取训练进度 (0-1)"""
        return min(1.0, self.get_total_samples() / self.samples_target)

    def save_checkpoint(self, filepath: str = None):
        """保存检查点 (所有实例的数据)"""
        if filepath is None:
            filepath = os.path.join(
                os.path.dirname(__file__), 'data',
                f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        all_samples = []
        for room_id, collector in self.collectors.items():
            for sample in collector.samples:
                sample_dict = sample.to_dict()
                sample_dict['room_id'] = room_id
                all_samples.append(sample_dict)

        data = {
            'num_instances': len(self.collectors),
            'total_samples': len(all_samples),
            'samples_target': self.samples_target,
            'samples': all_samples
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[AccelController] 检查点已保存: {filepath}")
        return filepath

    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for sample_dict in data['samples']:
            room_id = sample_dict.pop('room_id', 'default')
            if room_id not in self.collectors:
                self.collectors[room_id] = SimulationDataCollector()
            sample = DualFireSample.from_dict(sample_dict)
            self.collectors[room_id].samples.append(sample)

        print(f"[AccelController] 检查点已加载: {len(data['samples'])} 样本")


def select_fire_targets(observation: Dict, side: str = 'red',
                        min_angle_diff: float = 30) -> Optional[Tuple[int, int, int]]:
    """
    智能选择双机开火目标

    Args:
        observation: 态势数据
        side: 我方阵营
        min_angle_diff: 最小攻击角度差 (度)

    Returns:
        (shooter1_id, shooter2_id, target_id) 或 None
    """
    # 获取我方平台
    my_side = None
    enemy_tracks = []

    for side_data in observation.get('side_list', []):
        if side_data.get('side') == side:
            my_side = side_data
        else:
            # 敌方的track_list包含我们能看到的敌方目标
            pass

    if my_side is None:
        return None

    # 获取有弹药的平台
    available_shooters = []
    for p in my_side.get('platform_list', []):
        has_ammo = any(w.get('quantity', 0) > 0 for w in p.get('weapons', []))
        if has_ammo:
            available_shooters.append(p)

    if len(available_shooters) < 2:
        return None

    # 获取敌方目标 (非导弹)
    enemy_targets = []
    for t in my_side.get('track_list', []):
        if (t.get('platform_entity_side') != side and
            t.get('platform_entity_type') != '导弹'):
            enemy_targets.append(t)

    if not enemy_targets:
        return None

    # 随机选择目标
    target = random.choice(enemy_targets)
    t_lon, t_lat = target.get('longitude', 0), target.get('latitude', 0)

    # 选择攻击角度差最大的两架飞机
    best_pair = None
    best_angle_diff = 0

    for i, s1 in enumerate(available_shooters):
        for s2 in available_shooters[i+1:]:
            # 计算两架飞机相对目标的方位角
            bearing1 = math.degrees(math.atan2(
                s1['longitude'] - t_lon,
                s1['latitude'] - t_lat
            )) % 360

            bearing2 = math.degrees(math.atan2(
                s2['longitude'] - t_lon,
                s2['latitude'] - t_lat
            )) % 360

            angle_diff = abs(bearing1 - bearing2)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff

            if angle_diff > best_angle_diff:
                best_angle_diff = angle_diff
                best_pair = (s1['id'], s2['id'])

    if best_pair and best_angle_diff >= min_angle_diff:
        return (best_pair[0], best_pair[1], target['target_id'])

    return None


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("真实仿真数据采集器测试")
    print("=" * 60)

    # 创建采集器
    collector = SimulationDataCollector()

    print(f"\n采集器已创建")
    print(f"数据目录: {collector.data_dir}")
    print(f"自动保存间隔: {collector.auto_save_interval} 样本")

    # 模拟observation数据结构
    mock_observation = {
        'header': {'sim_time': 100.0},
        'side_list': [
            {
                'side': 'red',
                'platform_list': [
                    {
                        'id': 1, 'name': '红无人机1', 'type': '无人机',
                        'longitude': 145.9, 'latitude': 33.5, 'altitude': 5000,
                        'speed': 400, 'heading': 0.5,
                        'velocity_x': 200, 'velocity_y': 346, 'velocity_z': 0,
                        'weapons': [{'quantity': 2}]
                    },
                    {
                        'id': 2, 'name': '红无人机2', 'type': '无人机',
                        'longitude': 146.1, 'latitude': 33.3, 'altitude': 5500,
                        'speed': 400, 'heading': 2.5,
                        'velocity_x': -200, 'velocity_y': 346, 'velocity_z': 0,
                        'weapons': [{'quantity': 2}]
                    }
                ],
                'track_list': [
                    {
                        'target_id': 101, 'target_name': '蓝无人机1',
                        'platform_entity_type': '无人机',
                        'platform_entity_side': 'blue',
                        'longitude': 146.0, 'latitude': 33.4, 'altitude': 5200,
                        'speed': 350, 'heading': 1.0,
                        'v_x': 100, 'v_y': -300, 'v_z': 0,
                        'roll': 0.5
                    }
                ],
                'broken_list': []
            }
        ]
    }

    # 测试目标选择
    result = select_fire_targets(mock_observation, 'red')
    if result:
        print(f"\n选择的开火目标:")
        print(f"  射手1: {result[0]}")
        print(f"  射手2: {result[1]}")
        print(f"  目标: {result[2]}")
    else:
        print("\n未找到合适的开火目标")

    print("\n测试完成!")
