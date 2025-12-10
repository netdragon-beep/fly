"""
双机协同开火在线学习智能体

核心思想:
1. 边采集数据边更新模型 (Online Learning)
2. 使用 ε-greedy 策略平衡探索与利用
3. 经验回放 (Experience Replay) 提高样本效率
4. 定期评估模型性能并自适应调整探索率

优点:
- 实时学习，不需要等待采集完成
- 探索-利用平衡，避免陷入局部最优
- 可以在实战中持续改进
"""

import os
import sys
import math
import random
import json
import numpy as np
import torch
from collections import deque
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass

# 添加路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# 添加项目根目录到路径
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(_current_dir))))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from env.agent.agent_base import AutoAgentBase
    from utilities.yxScriptTreeFunc import YxScriptTreeFunc as decCmd
    from utilities.yxGeoUtils import YxGeoUtils
except ImportError:
    # 如果还是无法导入，尝试从当前目录的相对位置导入
    _agent_dir = os.path.dirname(os.path.dirname(os.path.dirname(_current_dir)))
    if _agent_dir not in sys.path:
        sys.path.insert(0, _agent_dir)
    from agent_base import AutoAgentBase
    # utilities应该在项目根目录
    from utilities.yxScriptTreeFunc import YxScriptTreeFunc as decCmd
    from utilities.yxGeoUtils import YxGeoUtils

# 导入双机开火模块
try:
    from .feature_extractor import DualFireFeatureExtractor, DualFireFeatures
    from .classifier import DualFireClassifier, MLPClassifier
    from .sim_data_collector import PendingFireEvent
    from .data_collector import DualFireSample
except ImportError:
    from feature_extractor import DualFireFeatureExtractor, DualFireFeatures
    from classifier import DualFireClassifier, MLPClassifier
    from sim_data_collector import PendingFireEvent
    from data_collector import DualFireSample


@dataclass
class OnlineConfig:
    """在线学习配置"""
    # 探索参数
    epsilon_start: float = 1.0       # 初始探索率
    epsilon_end: float = 0.1         # 最终探索率
    epsilon_decay: float = 0.995     # 探索率衰减

    # 学习参数
    batch_size: int = 32             # 在线更新批大小
    learning_rate: float = 1e-3      # 学习率
    update_interval: int = 10        # 每N个新样本更新一次
    min_samples_to_train: int = 100  # 开始训练的最小样本数

    # 经验回放
    replay_buffer_size: int = 10000  # 经验回放缓冲区大小
    priority_replay: bool = True     # 是否使用优先经验回放

    # 模型参数
    hidden_dims: Tuple[int, ...] = (128, 64, 32)

    # 评估参数
    eval_interval: int = 100         # 每N个样本评估一次
    confidence_threshold: float = 0.6  # 开火置信度阈值

    # GPU
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class PrioritizedReplayBuffer:
    """
    优先经验回放缓冲区

    根据TD误差或预测误差给样本分配优先级
    难以预测的样本会被更频繁地采样
    """

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        """
        Args:
            capacity: 缓冲区容量
            alpha: 优先级指数 (0=均匀采样, 1=完全按优先级)
            beta: 重要性采样指数
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001

        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def push(self, sample: DualFireSample, priority: float = None):
        """添加样本"""
        if priority is None:
            priority = self.max_priority

        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.position] = sample

        self.priorities[self.position] = priority ** self.alpha
        self.position = (self.position + 1) % self.capacity
        self.max_priority = max(self.max_priority, priority)

    def sample(self, batch_size: int) -> Tuple[List[DualFireSample], np.ndarray, np.ndarray]:
        """
        按优先级采样

        Returns:
            samples, indices, weights (用于重要性采样)
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        # 计算采样概率
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities / priorities.sum()

        # 采样
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]

        # 重要性采样权重
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()

        return samples, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (priority + 1e-6) ** self.alpha
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)


class OnlineDualFireAgent(AutoAgentBase):
    """
    双机协同开火在线学习智能体

    特点:
    1. ε-greedy 探索策略
    2. 在线更新模型
    3. 优先经验回放
    4. 自适应探索率
    """

    # 导弹参数
    MISSILE_SPEED = 1200
    MISSILE_TIMEOUT_FRAMES = 100
    MIN_FIRE_INTERVAL = 30

    def __init__(self, side: str, name: str,
                 config: OnlineConfig = None,
                 pretrained_model_path: str = None):
        """
        Args:
            side: 阵营
            name: 名称
            config: 在线学习配置
            pretrained_model_path: 预训练模型路径 (可选)
        """
        super().__init__(side, name)

        self.config = config or OnlineConfig()

        print(f"[OnlineDualFire] 初始化在线学习智能体")
        print(f"  设备: {self.config.device}")
        print(f"  初始探索率: {self.config.epsilon_start}")

        # 特征提取器
        self.feature_extractor = DualFireFeatureExtractor()

        # 在线模型
        self.model = MLPClassifier(
            input_dim=DualFireFeatures.feature_dim(),
            hidden_dims=self.config.hidden_dims
        ).to(self.config.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        self.criterion = torch.nn.BCELoss(reduction='none')  # 支持样本权重

        # 加载预训练模型 (如果有)
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            self._load_pretrained(pretrained_model_path)

        # 经验回放缓冲区
        if self.config.priority_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                self.config.replay_buffer_size
            )
        else:
            self.replay_buffer = deque(maxlen=self.config.replay_buffer_size)

        # 待定事件
        self.pending_events: List[PendingFireEvent] = []

        # 探索率
        self.epsilon = self.config.epsilon_start

        # 统计
        self.frame_count = 0
        self.last_fire_frame = 0
        self.total_fires = 0
        self.total_hits = 0
        self.total_samples = 0
        self.samples_since_update = 0

        # 训练历史
        self.train_losses = []
        self.eval_accuracies = []

        # 决策统计
        self.decisions = {'explore': 0, 'exploit': 0, 'model_fire': 0, 'model_wait': 0}

        # 基础战斗状态
        self.own_units = []
        self.enemy_units = []
        self.enemy_missiles = []
        self.current_actions = []
        self.commanded_units = set()
        self.broken_list = set()

        # 战场信息
        self.center_lat = (self.battlefield['min_lat'] + self.battlefield['max_lat']) / 2
        self.center_lon = (self.battlefield['min_lon'] + self.battlefield['max_lon']) / 2

        # 数据目录
        self.data_dir = os.path.join(_current_dir, 'data')
        self.model_dir = os.path.join(_current_dir, 'models')
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def _load_pretrained(self, model_path: str):
        """加载预训练模型"""
        try:
            checkpoint = torch.load(model_path, map_location=self.config.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[OnlineDualFire] 已加载预训练模型: {model_path}")
            # 预训练模型存在时，降低初始探索率
            self.config.epsilon_start = 0.3
        except Exception as e:
            print(f"[OnlineDualFire] 加载预训练模型失败: {e}")

    def update_decision(self, new_observation: Dict):
        """主决策函数"""
        self.observation = new_observation
        self.frame_count += 1
        self.current_actions = []
        self.commanded_units = set()

        # 解析态势
        self._parse_observation(new_observation)

        if not self.own_units:
            return []

        # 1. 更新待定事件 (判断命中/脱靶)
        self._update_pending_events()

        # 2. 在线学习更新
        if self.samples_since_update >= self.config.update_interval:
            self._online_update()

        # 3. 导弹规避
        self._evade_missiles()

        # 4. 双机协同开火决策 (ε-greedy)
        self._dual_fire_decision()

        # 5. 基础移动
        self._basic_movement()

        # 6. 定期评估和打印
        if self.frame_count % 100 == 0:
            self._print_statistics()

        if self.total_samples > 0 and self.total_samples % self.config.eval_interval == 0:
            self._evaluate_model()

        return self.current_actions

    def _parse_observation(self, obs):
        """解析态势"""
        assert obs.get('side') == self.side

        self.own_units = obs.get('platform_list', [])
        self.enemy_units = []
        self.enemy_missiles = []
        self.broken_list = set(obs.get('broken_list', []))

        for track in obs.get('track_list', []):
            if track.get('platform_entity_side') != self.side:
                if track.get('platform_entity_type') == '导弹':
                    self.enemy_missiles.append(track)
                else:
                    self.enemy_units.append(track)

    def _update_pending_events(self):
        """更新待定事件，收集样本"""
        resolved_events = []

        for event in self.pending_events:
            if event.resolved:
                continue

            frames_elapsed = self.frame_count - event.fire_time
            target_destroyed = event.target_name in self.broken_list

            # 检查导弹状态
            flying_missiles = {t.get('target_name', '') for t in self.observation.get('track_list', [])
                              if t.get('platform_entity_type') == '导弹'}
            missile1_flying = event.missile1_name in flying_missiles
            missile2_flying = event.missile2_name in flying_missiles

            # 判断结果
            if target_destroyed:
                if not missile1_flying:
                    event.hit_1 = 1
                if not missile2_flying:
                    event.hit_2 = 1
                event.resolved = True
                resolved_events.append(event)

            elif frames_elapsed > self.MISSILE_TIMEOUT_FRAMES:
                event.resolved = True
                resolved_events.append(event)

            elif not missile1_flying and not missile2_flying and not target_destroyed:
                event.resolved = True
                resolved_events.append(event)

        # 记录样本并添加到缓冲区
        for event in resolved_events:
            self._add_sample(event)
            self.pending_events.remove(event)

    def _add_sample(self, event: PendingFireEvent):
        """添加样本到经验回放缓冲区"""
        hit = 1 if (event.hit_1 or event.hit_2) else 0
        hit_count = event.hit_1 + event.hit_2

        sample = DualFireSample(
            features=event.features,
            hit=hit,
            hit_count=hit_count,
            timestamp=datetime.now().timestamp()
        )

        # 计算优先级 (预测误差)
        with torch.no_grad():
            X = torch.FloatTensor(event.features).unsqueeze(0).to(self.config.device)
            pred = self.model(X).item()
            priority = abs(pred - hit) + 0.1  # 预测误差作为优先级

        if self.config.priority_replay:
            self.replay_buffer.push(sample, priority)
        else:
            self.replay_buffer.append(sample)

        self.total_samples += 1
        self.samples_since_update += 1

        if hit:
            self.total_hits += 1

        # 打印
        result = "命中" if hit else "脱靶"
        print(f"[样本] {result} | 预测: {pred:.2f} | 误差: {priority:.2f} | "
              f"总样本: {self.total_samples}")

    def _online_update(self):
        """在线更新模型"""
        if len(self.replay_buffer) < self.config.min_samples_to_train:
            return

        self.model.train()

        # 从缓冲区采样
        if self.config.priority_replay:
            samples, indices, weights = self.replay_buffer.sample(self.config.batch_size)
            weights = torch.FloatTensor(weights).to(self.config.device)
        else:
            samples = random.sample(list(self.replay_buffer),
                                   min(self.config.batch_size, len(self.replay_buffer)))
            weights = torch.ones(len(samples)).to(self.config.device)
            indices = None

        # 准备数据
        X = torch.FloatTensor(np.array([s.features for s in samples])).to(self.config.device)
        y = torch.FloatTensor(np.array([s.hit for s in samples])).unsqueeze(1).to(self.config.device)

        # 前向传播
        self.optimizer.zero_grad()
        pred = self.model(X)

        # 加权损失
        loss = self.criterion(pred, y)
        weighted_loss = (loss * weights.unsqueeze(1)).mean()

        # 反向传播
        weighted_loss.backward()
        self.optimizer.step()

        # 更新优先级
        if self.config.priority_replay and indices is not None:
            with torch.no_grad():
                new_priorities = torch.abs(pred - y).squeeze().cpu().numpy()
                self.replay_buffer.update_priorities(indices, new_priorities)

        self.train_losses.append(weighted_loss.item())
        self.samples_since_update = 0

        # 衰减探索率
        self.epsilon = max(self.config.epsilon_end,
                          self.epsilon * self.config.epsilon_decay)

        if len(self.train_losses) % 10 == 0:
            avg_loss = np.mean(self.train_losses[-10:])
            print(f"[训练] Loss: {avg_loss:.4f} | ε: {self.epsilon:.3f} | "
                  f"缓冲区: {len(self.replay_buffer)}")

    def _dual_fire_decision(self):
        """双机协同开火决策 (ε-greedy)"""
        if self.frame_count - self.last_fire_frame < self.MIN_FIRE_INTERVAL:
            return

        if not self.enemy_units:
            return

        # 获取可用射手
        shooters = self._get_available_shooters()
        if len(shooters) < 2:
            return

        # 找到所有可能的双机+目标组合
        candidates = self._get_fire_candidates(shooters)
        if not candidates:
            return

        # ε-greedy 决策
        if random.random() < self.epsilon:
            # 探索: 随机选择一个组合并开火
            self.decisions['explore'] += 1
            chosen = random.choice(candidates)
            self._execute_dual_fire(*chosen)
        else:
            # 利用: 使用模型选择最佳组合
            self.decisions['exploit'] += 1
            best_candidate = None
            best_prob = -1

            self.model.eval()
            with torch.no_grad():
                for shooter1, shooter2, target, features in candidates:
                    X = torch.FloatTensor(features.to_array()).unsqueeze(0).to(self.config.device)
                    prob = self.model(X).item()

                    if prob > best_prob:
                        best_prob = prob
                        best_candidate = (shooter1, shooter2, target, features)

            # 只有当模型置信度足够高时才开火
            if best_candidate and best_prob >= self.config.confidence_threshold:
                self.decisions['model_fire'] += 1
                self._execute_dual_fire(*best_candidate)
                print(f"[模型决策] 开火! 置信度: {best_prob:.2f}")
            else:
                self.decisions['model_wait'] += 1
                if best_candidate:
                    print(f"[模型决策] 等待 | 最高置信度: {best_prob:.2f} < {self.config.confidence_threshold}")

    def _get_fire_candidates(self, shooters: List[Dict]) -> List[Tuple]:
        """获取所有可能的开火组合"""
        candidates = []

        for target in self.enemy_units:
            target_id = target.get('target_id', target.get('id'))

            # 检查目标是否已有导弹
            pending_to_target = sum(1 for e in self.pending_events if e.target_id == target_id)
            if pending_to_target >= 2:
                continue

            for i, s1 in enumerate(shooters):
                for s2 in shooters[i+1:]:
                    features = self.feature_extractor.extract(s1, s2, target)

                    # 基本过滤: 距离太远不考虑
                    avg_dist = (features.dist_1 + features.dist_2) / 2
                    if avg_dist > 35000:
                        continue

                    candidates.append((s1, s2, target, features))

        return candidates

    def _get_available_shooters(self) -> List[Dict]:
        """获取有弹药的射手"""
        shooters = []
        for unit in self.own_units:
            if unit['name'] in self.commanded_units:
                continue
            has_ammo = any(w.get('quantity', 0) > 0 for w in unit.get('weapons', []))
            if has_ammo:
                shooters.append(unit)
        return shooters

    def _execute_dual_fire(self, shooter1: Dict, shooter2: Dict,
                           target: Dict, features: DualFireFeatures):
        """执行双机协同开火"""
        target_id = target.get('target_id', target.get('id'))
        target_name = target.get('target_name', target.get('name', str(target_id)))

        avg_dist = (features.dist_1 + features.dist_2) / 2
        expected_tof = avg_dist / self.MISSILE_SPEED

        missile1_name = f"{shooter1['name']}_导弹_{self._get_missile_num(shooter1)}"
        missile2_name = f"{shooter2['name']}_导弹_{self._get_missile_num(shooter2)}"

        event = PendingFireEvent(
            fire_time=self.frame_count,
            shooter1_id=shooter1['id'],
            shooter2_id=shooter2['id'],
            target_id=target_id,
            target_name=target_name,
            missile1_name=missile1_name,
            missile2_name=missile2_name,
            features=features.to_array(),
            expected_tof=expected_tof
        )

        if self._try_fire_weapon(shooter1) and self._try_fire_weapon(shooter2):
            self.current_actions.append(decCmd.fire_track(shooter1['name'], target_name))
            self.current_actions.append(decCmd.fire_track(shooter2['name'], target_name))

            self.pending_events.append(event)
            self.total_fires += 1
            self.last_fire_frame = self.frame_count

            print(f"\n[双机开火] Frame {self.frame_count} | ε={self.epsilon:.3f}")
            print(f"  射手: {shooter1['name']} + {shooter2['name']}")
            print(f"  目标: {target_name}")
            print(f"  角度差: {features.angle_diff:.1f}° | 距离: {avg_dist/1000:.1f}km")

    def _evade_missiles(self):
        """导弹规避"""
        for missile in self.enemy_missiles:
            m_lon = missile.get('longitude', 0)
            m_lat = missile.get('latitude', 0)
            m_heading = math.degrees(missile.get('heading', 0)) % 360

            for unit in self.own_units:
                if unit['name'] in self.commanded_units:
                    continue

                u_lon = unit.get('longitude', 0)
                u_lat = unit.get('latitude', 0)

                dist = YxGeoUtils.haversine_distance(u_lon, u_lat, m_lon, m_lat)
                bearing = YxGeoUtils.calculate_bearing(m_lon, m_lat, u_lon, u_lat)
                angle_diff = abs((bearing - m_heading + 180) % 360 - 180)

                if dist < 8000 and angle_diff < 20:
                    evade_dir = (m_heading + 90) % 360
                    lon_off, lat_off = YxGeoUtils.km_to_lon_lat(self.center_lat, 3, evade_dir)
                    evade_pt = (u_lat + lat_off, u_lon + lon_off, unit.get('altitude', 3000))
                    self.current_actions.append(decCmd.fly_to_point(unit['name'], evade_pt, 550))
                    self.commanded_units.add(unit['name'])

    def _basic_movement(self):
        """基础移动"""
        for unit in self.own_units:
            if unit['name'] in self.commanded_units:
                continue

            if not self.enemy_units:
                continue

            u_lon = unit.get('longitude', 0)
            u_lat = unit.get('latitude', 0)

            closest = min(self.enemy_units, key=lambda e:
                YxGeoUtils.haversine_distance(u_lon, u_lat,
                    e.get('longitude', 0), e.get('latitude', 0)))

            e_lon = closest.get('longitude', 0)
            e_lat = closest.get('latitude', 0)
            e_alt = closest.get('altitude', 3000)

            dist = YxGeoUtils.haversine_distance(u_lon, u_lat, e_lon, e_lat)

            if dist > 15000:
                target_pt = (e_lat, e_lon, e_alt)
                self.current_actions.append(decCmd.fly_to_point(unit['name'], target_pt, 500))
                self.commanded_units.add(unit['name'])

    def _get_missile_num(self, platform: Dict) -> int:
        max_ammo = 4 if platform.get('type') == '有人机' else 2
        current_ammo = sum(w.get('quantity', 0) for w in platform.get('weapons', []))
        return max_ammo - current_ammo + 1

    def _try_fire_weapon(self, unit: Dict) -> bool:
        for weapon in unit.get('weapons', []):
            if weapon.get('quantity', 0) > 0:
                weapon['quantity'] -= 1
                return True
        return False

    def _evaluate_model(self):
        """评估模型性能"""
        if len(self.replay_buffer) < 100:
            return

        self.model.eval()

        # 从缓冲区获取所有样本评估
        if self.config.priority_replay:
            samples = self.replay_buffer.buffer
        else:
            samples = list(self.replay_buffer)

        X = np.array([s.features for s in samples])
        y = np.array([s.hit for s in samples])

        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.config.device)
            pred = self.model(X_t).cpu().numpy().flatten()

        pred_labels = (pred > 0.5).astype(int)
        accuracy = (pred_labels == y).mean()

        self.eval_accuracies.append(accuracy)

        hit_rate = y.mean()
        pred_hit_rate = pred_labels.mean()

        print(f"\n[评估] 样本: {len(samples)} | 准确率: {accuracy:.2%}")
        print(f"  实际命中率: {hit_rate:.2%} | 预测命中率: {pred_hit_rate:.2%}")
        print(f"  探索: {self.decisions['explore']} | 利用: {self.decisions['exploit']}")
        print(f"  模型开火: {self.decisions['model_fire']} | 模型等待: {self.decisions['model_wait']}")

    def _print_statistics(self):
        """打印统计"""
        hit_rate = self.total_hits / self.total_fires if self.total_fires > 0 else 0
        print(f"\n[统计] Frame {self.frame_count} | ε={self.epsilon:.3f}")
        print(f"  样本: {self.total_samples} | 开火: {self.total_fires} | 命中率: {hit_rate:.1%}")
        print(f"  缓冲区: {len(self.replay_buffer)} | 待定: {len(self.pending_events)}")

    def save_model(self, filepath: str = None):
        """保存模型"""
        if filepath is None:
            filepath = os.path.join(self.model_dir,
                f"dual_fire_online_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_samples': self.total_samples,
            'train_losses': self.train_losses,
            'eval_accuracies': self.eval_accuracies,
            'config': self.config.__dict__
        }, filepath)

        print(f"[保存] 模型已保存: {filepath}")
        return filepath

    def save_data(self, filepath: str = None):
        """保存采集的数据"""
        if self.config.priority_replay:
            samples = self.replay_buffer.buffer
        else:
            samples = list(self.replay_buffer)

        if not samples:
            return None

        if filepath is None:
            filepath = os.path.join(self.data_dir,
                f"dual_fire_online_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        data = {
            'source': 'online_learning',
            'num_samples': len(samples),
            'total_fires': self.total_fires,
            'total_hits': self.total_hits,
            'epsilon_final': self.epsilon,
            'samples': [{'features': s.features.tolist(), 'hit': s.hit,
                        'hit_count': s.hit_count, 'timestamp': s.timestamp}
                       for s in samples]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[保存] 数据已保存: {filepath} ({len(samples)} 样本)")
        return filepath


# 导出
__all__ = ['OnlineDualFireAgent', 'OnlineConfig', 'PrioritizedReplayBuffer']
