"""
双机协同开火数据采集智能体

专门用于在真实仿真环境中采集双机协同开火数据
可以独立运行，也可以集成到现有策略树中

使用方法:
1. 独立运行: 替换原有Agent进行数据采集
2. 集成模式: 在攻击逻辑中调用采集器

采集策略:
- 随机选择两架有弹药的飞机
- 选择角度差较大的目标组合 (利于训练)
- 记录开火时的特征和命中结果
"""

import os
import sys
import math
import random
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime

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
    _agent_dir = os.path.dirname(os.path.dirname(os.path.dirname(_current_dir)))
    if _agent_dir not in sys.path:
        sys.path.insert(0, _agent_dir)
    from agent_base import AutoAgentBase
    from utilities.yxScriptTreeFunc import YxScriptTreeFunc as decCmd
    from utilities.yxGeoUtils import YxGeoUtils

# 导入双机开火模块
try:
    from .feature_extractor import DualFireFeatureExtractor, DualFireFeatures
    from .sim_data_collector import SimulationDataCollector, PendingFireEvent
except ImportError:
    from feature_extractor import DualFireFeatureExtractor, DualFireFeatures
    from sim_data_collector import SimulationDataCollector, PendingFireEvent


class DualFireDataCollectionAgent(AutoAgentBase):
    """
    双机协同开火数据采集智能体

    专门用于采集训练数据:
    1. 随机选择双机组合和目标
    2. 执行双机协同开火
    3. 跟踪命中结果
    4. 保存数据用于训练分类器
    """

    # 采集参数
    MIN_FIRE_INTERVAL = 30           # 最小开火间隔 (帧)
    MIN_ANGLE_DIFF = 20              # 最小角度差 (度) - 放宽以采集更多数据
    FIRE_PROBABILITY = 0.5           # 每次满足条件时开火的概率
    MAX_PENDING_PER_TARGET = 2       # 每个目标最多待定导弹数

    # 导弹参数
    MISSILE_SPEED = 1200             # m/s
    MISSILE_TIMEOUT_FRAMES = 100     # 超时帧数

    def __init__(self, side, name, auto_save_interval: int = 50):
        """
        Args:
            side: 阵营
            name: 名称
            auto_save_interval: 每N个样本自动保存
        """
        super().__init__(side, name)

        print(f"[DualFireCollector] 初始化数据采集智能体 side={side}")

        # 数据采集器
        self.feature_extractor = DualFireFeatureExtractor()
        self.samples = []
        self.pending_events: List[PendingFireEvent] = []

        # 状态跟踪
        self.frame_count = 0
        self.last_fire_frame = 0
        self.total_fires = 0
        self.total_hits = 0

        # 自动保存
        self.auto_save_interval = auto_save_interval
        self._samples_since_save = 0

        # 数据目录
        self.data_dir = os.path.join(_current_dir, 'data')
        os.makedirs(self.data_dir, exist_ok=True)

        # 基础战斗状态
        self.own_units = []
        self.enemy_units = []
        self.enemy_missiles = []
        self.current_actions = []
        self.commanded_units = set()

        # 战场信息
        self.center_lat = (self.battlefield['min_lat'] + self.battlefield['max_lat']) / 2
        self.center_lon = (self.battlefield['min_lon'] + self.battlefield['max_lon']) / 2

        print(f"[DualFireCollector] 数据目录: {self.data_dir}")

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

        # 1. 更新待定事件状态 (检查命中)
        self._update_pending_events()

        # 2. 导弹规避 (保护自己)
        self._evade_missiles()

        # 3. 尝试双机协同开火 (采集数据)
        self._try_dual_fire()

        # 4. 基础移动 (接近敌人)
        self._basic_movement()

        # 5. 定期打印统计
        if self.frame_count % 100 == 0:
            self._print_statistics()

        return self.current_actions

    def _parse_observation(self, obs):
        """解析态势数据"""
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
        """更新待定事件状态，判断命中/脱靶"""
        resolved_events = []

        for event in self.pending_events:
            if event.resolved:
                continue

            frames_elapsed = self.frame_count - event.fire_time

            # 检查目标是否被击毁
            target_destroyed = event.target_name in self.broken_list

            # 检查导弹是否还在飞
            missile1_alive = event.missile1_name not in self.broken_list
            missile2_alive = event.missile2_name not in self.broken_list

            # 检查导弹是否在track_list中 (还在飞行)
            flying_missiles = {t.get('target_name', '') for t in self.observation.get('track_list', [])
                              if t.get('platform_entity_type') == '导弹'}
            missile1_flying = event.missile1_name in flying_missiles
            missile2_flying = event.missile2_name in flying_missiles

            # 判断命中
            if target_destroyed:
                # 目标被毁，检查是哪发导弹命中
                if not missile1_flying:
                    event.hit_1 = 1
                if not missile2_flying:
                    event.hit_2 = 1
                event.resolved = True
                resolved_events.append(event)
                print(f"[命中!] 目标 {event.target_name} 被击毁, "
                      f"导弹1: {'命中' if event.hit_1 else '?'}, "
                      f"导弹2: {'命中' if event.hit_2 else '?'}")

            elif frames_elapsed > self.MISSILE_TIMEOUT_FRAMES:
                # 超时，认为脱靶
                event.resolved = True
                resolved_events.append(event)
                print(f"[脱靶] 目标 {event.target_name} 导弹超时")

            elif not missile1_flying and not missile2_flying and not target_destroyed:
                # 两发导弹都消失但目标未被击毁 = 脱靶
                event.resolved = True
                resolved_events.append(event)
                print(f"[脱靶] 目标 {event.target_name} 导弹消失但未命中")

        # 记录已确定结果的样本
        for event in resolved_events:
            self._record_sample(event)
            self.pending_events.remove(event)

    def _record_sample(self, event: PendingFireEvent):
        """记录一条样本"""
        try:
            from .data_collector import DualFireSample
        except ImportError:
            from data_collector import DualFireSample

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

        # 自动保存
        if self._samples_since_save >= self.auto_save_interval:
            self.save_data()
            self._samples_since_save = 0

    def _try_dual_fire(self):
        """尝试双机协同开火"""
        # 检查冷却
        if self.frame_count - self.last_fire_frame < self.MIN_FIRE_INTERVAL:
            return

        # 没有敌人
        if not self.enemy_units:
            return

        # 获取有弹药的单位
        shooters = self._get_available_shooters()
        if len(shooters) < 2:
            return

        # 随机决定是否开火 (增加数据多样性)
        if random.random() > self.FIRE_PROBABILITY:
            return

        # 选择最佳双机组合和目标
        best_combo = self._select_best_fire_combo(shooters)
        if best_combo is None:
            return

        shooter1, shooter2, target, features = best_combo

        # 检查目标是否已有太多导弹
        target_id = target.get('target_id', target.get('id'))
        pending_to_target = sum(1 for e in self.pending_events if e.target_id == target_id)
        if pending_to_target >= self.MAX_PENDING_PER_TARGET:
            return

        # 执行双机开火
        self._execute_dual_fire(shooter1, shooter2, target, features)

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

    def _select_best_fire_combo(self, shooters: List[Dict]) -> Optional[Tuple]:
        """选择最佳的双机+目标组合"""
        best_combo = None
        best_score = -1

        for target in self.enemy_units:
            t_lon = target.get('longitude', 0)
            t_lat = target.get('latitude', 0)

            for i, s1 in enumerate(shooters):
                for s2 in shooters[i+1:]:
                    # 计算角度差
                    bearing1 = self._calc_bearing(t_lon, t_lat,
                                                  s1.get('longitude', 0), s1.get('latitude', 0))
                    bearing2 = self._calc_bearing(t_lon, t_lat,
                                                  s2.get('longitude', 0), s2.get('latitude', 0))

                    angle_diff = abs(bearing1 - bearing2)
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff

                    # 计算距离
                    dist1 = YxGeoUtils.haversine_distance(
                        s1.get('longitude', 0), s1.get('latitude', 0),
                        t_lon, t_lat
                    )
                    dist2 = YxGeoUtils.haversine_distance(
                        s2.get('longitude', 0), s2.get('latitude', 0),
                        t_lon, t_lat
                    )
                    avg_dist = (dist1 + dist2) / 2

                    # 评分: 角度差大 + 距离适中
                    score = angle_diff / 90.0  # 角度差评分

                    # 距离惩罚
                    if avg_dist > 30000:
                        score *= 0.3
                    elif avg_dist > 25000:
                        score *= 0.6
                    elif avg_dist < 8000:
                        score *= 0.8

                    # 随机扰动 (增加数据多样性)
                    score *= random.uniform(0.8, 1.2)

                    if score > best_score:
                        best_score = score
                        # 提取特征
                        features = self.feature_extractor.extract(s1, s2, target)
                        best_combo = (s1, s2, target, features)

        # 检查角度差是否满足最小要求
        if best_combo and best_combo[3].angle_diff >= self.MIN_ANGLE_DIFF:
            return best_combo

        return None

    def _execute_dual_fire(self, shooter1: Dict, shooter2: Dict, target: Dict,
                           features: DualFireFeatures):
        """执行双机协同开火"""
        target_id = target.get('target_id', target.get('id'))
        target_name = target.get('target_name', target.get('name', str(target_id)))

        # 计算预计飞行时间
        avg_dist = (features.dist_1 + features.dist_2) / 2
        expected_tof = avg_dist / self.MISSILE_SPEED

        # 生成导弹名称
        missile1_name = f"{shooter1['name']}_导弹_{self._get_missile_num(shooter1)}"
        missile2_name = f"{shooter2['name']}_导弹_{self._get_missile_num(shooter2)}"

        # 创建待定事件
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

        # 扣除弹药并发射
        if self._try_fire_weapon(shooter1) and self._try_fire_weapon(shooter2):
            self.current_actions.append(decCmd.fire_track(shooter1['name'], target_name))
            self.current_actions.append(decCmd.fire_track(shooter2['name'], target_name))

            self.pending_events.append(event)
            self.total_fires += 1
            self.last_fire_frame = self.frame_count

            print(f"\n[双机开火] Frame {self.frame_count}")
            print(f"  射手: {shooter1['name']} + {shooter2['name']}")
            print(f"  目标: {target_name}")
            print(f"  角度差: {features.angle_diff:.1f}°")
            print(f"  距离: {avg_dist/1000:.1f}km")
            print(f"  样本总数: {len(self.samples)}, 待定: {len(self.pending_events)}")

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

                # 导弹正在接近
                if dist < 8000 and angle_diff < 20:
                    # 垂直于导弹方向规避
                    evade_dir = (m_heading + 90) % 360
                    lon_off, lat_off = YxGeoUtils.km_to_lon_lat(self.center_lat, 3, evade_dir)

                    evade_pt = (u_lat + lat_off, u_lon + lon_off, unit.get('altitude', 3000))
                    self.current_actions.append(decCmd.fly_to_point(unit['name'], evade_pt, 550))
                    self.commanded_units.add(unit['name'])

    def _basic_movement(self):
        """基础移动 - 向敌人接近"""
        for unit in self.own_units:
            if unit['name'] in self.commanded_units:
                continue

            if not self.enemy_units:
                continue

            # 找最近敌人
            u_lon = unit.get('longitude', 0)
            u_lat = unit.get('latitude', 0)

            closest = min(self.enemy_units, key=lambda e:
                YxGeoUtils.haversine_distance(u_lon, u_lat,
                    e.get('longitude', 0), e.get('latitude', 0)))

            e_lon = closest.get('longitude', 0)
            e_lat = closest.get('latitude', 0)
            e_alt = closest.get('altitude', 3000)

            dist = YxGeoUtils.haversine_distance(u_lon, u_lat, e_lon, e_lat)

            # 距离太远就接近
            if dist > 15000:
                target_pt = (e_lat, e_lon, e_alt)
                self.current_actions.append(decCmd.fly_to_point(unit['name'], target_pt, 500))
                self.commanded_units.add(unit['name'])

    def _calc_bearing(self, from_lon, from_lat, to_lon, to_lat) -> float:
        """计算方位角"""
        dx = to_lon - from_lon
        dy = to_lat - from_lat
        return math.degrees(math.atan2(dx, dy)) % 360

    def _get_missile_num(self, platform: Dict) -> int:
        """获取导弹编号"""
        max_ammo = 4 if platform.get('type') == '有人机' else 2
        current_ammo = sum(w.get('quantity', 0) for w in platform.get('weapons', []))
        return max_ammo - current_ammo + 1

    def _try_fire_weapon(self, unit: Dict) -> bool:
        """尝试扣除弹药"""
        for weapon in unit.get('weapons', []):
            if weapon.get('quantity', 0) > 0:
                weapon['quantity'] -= 1
                return True
        return False

    def _print_statistics(self):
        """打印统计信息"""
        hit_rate = self.total_hits / self.total_fires if self.total_fires > 0 else 0
        print(f"\n[统计] Frame {self.frame_count}")
        print(f"  样本: {len(self.samples)}, 待定: {len(self.pending_events)}")
        print(f"  开火: {self.total_fires}, 命中: {self.total_hits}, 命中率: {hit_rate:.1%}")

    def save_data(self, filename: str = None):
        """保存采集数据"""
        if not self.samples:
            print("[保存] 没有数据")
            return None

        if filename is None:
            filename = f"dual_fire_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = os.path.join(self.data_dir, filename)

        data = {
            'source': 'simulation',
            'num_samples': len(self.samples),
            'feature_dim': DualFireFeatures.feature_dim(),
            'feature_names': DualFireFeatures.feature_names(),
            'total_fires': self.total_fires,
            'total_hits': self.total_hits,
            'samples': [{'features': s.features.tolist(), 'hit': s.hit,
                        'hit_count': s.hit_count, 'timestamp': s.timestamp}
                       for s in self.samples]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[保存] {filepath} ({len(self.samples)} 样本)")
        return filepath

    def get_training_data(self):
        """获取训练数据"""
        import numpy as np
        if not self.samples:
            return None, None
        X = np.array([s.features for s in self.samples])
        y = np.array([s.hit for s in self.samples])
        return X, y


# 导出
__all__ = ['DualFireDataCollectionAgent']
