"""
双机协同开火行为树节点

集成到策略树中，用于:
1. 训练模式: 随机开火采集数据
2. 实战模式: 使用训练好的分类器智能决策

使用方法:
在策略树中添加 DualFireAction 节点
"""

import os
import random
import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    from .feature_extractor import DualFireFeatureExtractor, DualFireFeatures
    from .classifier import DualFireClassifier
    from .sim_data_collector import SimulationDataCollector, select_fire_targets, PendingFireEvent
except ImportError:
    from feature_extractor import DualFireFeatureExtractor, DualFireFeatures
    from classifier import DualFireClassifier
    from sim_data_collector import SimulationDataCollector, select_fire_targets, PendingFireEvent


class DualFireAction:
    """
    双机协同开火行为树节点

    支持两种模式:
    1. 训练模式 (training=True): 随机开火，采集数据
    2. 实战模式 (training=False): 使用分类器决策
    """

    def __init__(self, side: str = 'red',
                 training: bool = False,
                 model_path: str = None,
                 model_type: str = 'mlp',
                 confidence_threshold: float = 0.6,
                 min_fire_interval: float = 5.0,
                 min_angle_diff: float = 30.0):
        """
        Args:
            side: 控制的阵营
            training: 是否为训练模式
            model_path: 分类器模型路径 (实战模式需要)
            model_type: 分类器类型
            confidence_threshold: 开火置信度阈值
            min_fire_interval: 最小开火间隔 (秒)
            min_angle_diff: 最小攻击角度差 (度)
        """
        self.side = side
        self.training = training
        self.confidence_threshold = confidence_threshold
        self.min_fire_interval = min_fire_interval
        self.min_angle_diff = min_angle_diff

        self.feature_extractor = DualFireFeatureExtractor()
        self.classifier = None
        self.collector = None

        # 状态跟踪
        self.last_fire_time = 0
        self._fire_cooldowns: Dict[int, float] = {}  # platform_id -> last_fire_time

        # 加载模型 (实战模式)
        if not training:
            self._load_model(model_path, model_type)

        # 数据采集器 (训练模式)
        if training:
            self.collector = SimulationDataCollector()

    def _load_model(self, model_path: str, model_type: str):
        """加载分类器模型"""
        if model_path is None:
            # 使用默认路径
            model_dir = os.path.join(os.path.dirname(__file__), 'models')
            model_ext = '.pt' if model_type == 'mlp' else '.pkl'
            model_path = os.path.join(model_dir, f"dual_fire_{model_type}_best{model_ext}")

        if os.path.exists(model_path):
            self.classifier = DualFireClassifier(model_type=model_type)
            self.classifier.load(model_path)
            print(f"[DualFireAction] 已加载模型: {model_path}")
        else:
            print(f"[DualFireAction] 警告: 模型不存在: {model_path}")
            print("  将使用规则决策作为后备")

    def execute(self, observation: Dict, fun_tool, sim_time: float = None) -> List[Dict]:
        """
        执行双机协同开火决策

        Args:
            observation: 态势数据 (完整observation或side数据)
            fun_tool: YxScriptTreeFunc 实例
            sim_time: 当前仿真时间

        Returns:
            开火指令列表
        """
        if sim_time is None:
            sim_time = observation.get('header', {}).get('sim_time', 0)

        # 获取我方数据
        side_data = self._get_side_data(observation)
        if side_data is None:
            return []

        # 更新数据采集器 (训练模式)
        if self.training and self.collector:
            self.collector.update(observation, sim_time)

        # 检查开火间隔
        if sim_time - self.last_fire_time < self.min_fire_interval:
            return []

        # 查找开火机会
        fire_opportunity = self._find_fire_opportunity(side_data, sim_time)
        if fire_opportunity is None:
            return []

        shooter1, shooter2, target, features = fire_opportunity

        # 决策是否开火
        should_fire, confidence, reason = self._should_fire(features)

        if not should_fire:
            return []

        # 执行开火
        cmds = self._execute_fire(shooter1, shooter2, target, features, fun_tool, sim_time)

        if cmds:
            self.last_fire_time = sim_time
            print(f"[DualFireAction] {reason}")
            print(f"  射手: {shooter1['name']} + {shooter2['name']}")
            print(f"  目标: {target.get('target_name', target.get('target_id'))}")
            print(f"  角度差: {features.angle_diff:.1f}°")

        return cmds

    def _get_side_data(self, observation: Dict) -> Optional[Dict]:
        """获取我方阵营数据"""
        if 'platform_list' in observation:
            return observation
        elif 'side_list' in observation:
            for side in observation['side_list']:
                if side.get('side') == self.side:
                    return side
        return None

    def _find_fire_opportunity(self, side_data: Dict, sim_time: float) -> Optional[Tuple]:
        """
        查找开火机会

        Returns:
            (shooter1, shooter2, target, features) 或 None
        """
        # 获取有弹药且未在冷却的平台
        available_shooters = []
        for p in side_data.get('platform_list', []):
            has_ammo = any(w.get('quantity', 0) > 0 for w in p.get('weapons', []))
            cooldown_ok = sim_time - self._fire_cooldowns.get(p['id'], 0) >= self.min_fire_interval
            if has_ammo and cooldown_ok:
                available_shooters.append(p)

        if len(available_shooters) < 2:
            return None

        # 获取敌方目标
        enemy_targets = []
        for t in side_data.get('track_list', []):
            if (t.get('platform_entity_side') != self.side and
                t.get('platform_entity_type') != '导弹'):
                enemy_targets.append(t)

        if not enemy_targets:
            return None

        # 训练模式: 随机选择
        if self.training:
            target = random.choice(enemy_targets)
            shooters = random.sample(available_shooters, 2)
            shooter1, shooter2 = shooters[0], shooters[1]
        else:
            # 实战模式: 选择最佳组合
            best_opportunity = None
            best_score = -1

            for target in enemy_targets:
                for i, s1 in enumerate(available_shooters):
                    for s2 in available_shooters[i+1:]:
                        features = self.feature_extractor.extract(s1, s2, target)

                        # 计算得分 (角度差越大越好)
                        score = features.angle_diff / 90.0

                        # 距离惩罚 (太远减分)
                        avg_dist = (features.dist_1 + features.dist_2) / 2
                        if avg_dist > 25000:
                            score *= 0.5

                        if score > best_score:
                            best_score = score
                            best_opportunity = (s1, s2, target, features)

            if best_opportunity is None:
                return None

            shooter1, shooter2, target, features = best_opportunity

            # 检查角度差是否满足要求
            if features.angle_diff < self.min_angle_diff:
                return None

            return best_opportunity

        # 训练模式: 提取特征
        features = self.feature_extractor.extract(shooter1, shooter2, target)
        return (shooter1, shooter2, target, features)

    def _should_fire(self, features: DualFireFeatures) -> Tuple[bool, float, str]:
        """
        决策是否应该开火

        Returns:
            (should_fire, confidence, reason)
        """
        # 训练模式: 随机决策 (以采集各种情况的数据)
        if self.training:
            # 70%概率开火
            should_fire = random.random() < 0.7
            return should_fire, 0.7, "训练模式随机开火"

        # 使用分类器
        if self.classifier and self.classifier.is_trained:
            X = features.to_array().reshape(1, -1)
            _, probs = self.classifier.predict_batch(X)
            prob = probs[0]

            should_fire = prob >= self.confidence_threshold
            reason = f"分类器预测: {prob:.0%} ({'开火' if should_fire else '等待'})"
            return should_fire, prob, reason

        # 后备: 规则决策
        return self._rule_based_decision(features)

    def _rule_based_decision(self, features: DualFireFeatures) -> Tuple[bool, float, str]:
        """规则决策 (后备方案)"""
        reasons = []
        score = 0.5

        # 角度差加分
        if features.angle_diff >= 60:
            score += 0.2
            reasons.append(f"角度差大({features.angle_diff:.0f}°)")
        elif features.angle_diff < 30:
            score -= 0.2
            reasons.append(f"角度差小({features.angle_diff:.0f}°)")

        # 距离
        avg_dist = (features.dist_1 + features.dist_2) / 2
        if avg_dist < 15000:
            score += 0.1
            reasons.append("距离近")
        elif avg_dist > 25000:
            score -= 0.1
            reasons.append("距离远")

        # 横向速度差异
        lateral_diff = abs(features.target_v_lateral_1 - features.target_v_lateral_2)
        if lateral_diff > 100:
            score += 0.1
            reasons.append("目标难以同时躲避")

        should_fire = score >= self.confidence_threshold
        reason = f"规则决策: {score:.0%} - " + ", ".join(reasons) if reasons else f"规则决策: {score:.0%}"

        return should_fire, score, reason

    def _execute_fire(self, shooter1: Dict, shooter2: Dict, target: Dict,
                      features: DualFireFeatures, fun_tool, sim_time: float) -> List[Dict]:
        """执行开火"""
        shooter1_id = shooter1['id']
        shooter2_id = shooter2['id']
        target_id = target.get('target_id')

        # 更新冷却时间
        self._fire_cooldowns[shooter1_id] = sim_time
        self._fire_cooldowns[shooter2_id] = sim_time

        # 训练模式: 记录待定事件
        if self.training and self.collector:
            avg_dist = (features.dist_1 + features.dist_2) / 2
            event = PendingFireEvent(
                fire_time=sim_time,
                shooter1_id=shooter1_id,
                shooter2_id=shooter2_id,
                target_id=target_id,
                target_name=target.get('target_name', str(target_id)),
                missile1_name=f"{shooter1['name']}_导弹_{self._get_missile_num(shooter1)}",
                missile2_name=f"{shooter2['name']}_导弹_{self._get_missile_num(shooter2)}",
                features=features.to_array(),
                expected_tof=avg_dist / 1200
            )
            self.collector.pending_events.append(event)
            self.collector.total_fires += 1

        # 调用仿真接口开火
        try:
            fun_tool.attack_target(shooter1_id, target_id)
            fun_tool.attack_target(shooter2_id, target_id)
            return [{'type': 'dual_fire', 'shooters': [shooter1_id, shooter2_id], 'target': target_id}]
        except Exception as e:
            print(f"[DualFireAction] 开火失败: {e}")
            return []

    def _get_missile_num(self, platform: Dict) -> int:
        """获取导弹编号"""
        max_ammo = 4 if platform.get('type') == '有人机' else 2
        current_ammo = sum(w.get('quantity', 0) for w in platform.get('weapons', []))
        return max_ammo - current_ammo + 1

    def get_statistics(self) -> Dict:
        """获取统计信息 (训练模式)"""
        if self.collector:
            return self.collector.get_statistics()
        return {}

    def save_data(self, filename: str = None) -> Optional[str]:
        """保存采集数据 (训练模式)"""
        if self.collector:
            return self.collector.save(filename)
        return None


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("双机协同开火行为树节点测试")
    print("=" * 60)

    # 创建节点 (训练模式)
    action = DualFireAction(side='red', training=True)

    # 模拟态势数据
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

    # 测试查找开火机会
    side_data = mock_observation['side_list'][0]
    opportunity = action._find_fire_opportunity(side_data, 100.0)

    if opportunity:
        shooter1, shooter2, target, features = opportunity
        print(f"\n找到开火机会:")
        print(f"  射手1: {shooter1['name']}")
        print(f"  射手2: {shooter2['name']}")
        print(f"  目标: {target['target_name']}")
        print(f"  角度差: {features.angle_diff:.1f}°")

        # 测试决策
        should_fire, confidence, reason = action._should_fire(features)
        print(f"\n决策结果:")
        print(f"  开火: {should_fire}")
        print(f"  置信度: {confidence:.0%}")
        print(f"  原因: {reason}")
    else:
        print("\n未找到开火机会")

    print("\n测试完成!")
