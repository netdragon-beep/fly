"""
双机协同开火特征提取器

从态势数据中提取双机协同攻击的关键特征
用于分类器判断是否应该开火

核心修正:
- 目标相对于每架射手的逃逸速度是不同的!
- 射手1看到的目标横向速度 != 射手2看到的横向速度
- 因为目标相对于两架射手的方向不同
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DualFireFeatures:
    """
    双机协同开火特征 (19维)

    核心思想: 目标相对于每架射手的逃逸速度是不同的!
    """
    # ===== 射手1相对目标的特征 (6维) =====
    dist_1: float               # 射手1到目标距离 (m)
    aspect_1: float             # 射手1的姿态角 (0=尾追, 180=迎头)
    closure_rate_1: float       # 射手1的接近率 (m/s)
    off_boresight_1: float      # 射手1的离轴角 (度)
    target_v_lateral_1: float   # 目标相对射手1的横向逃逸速度 (m/s) - 关键!
    target_v_radial_1: float    # 目标相对射手1的径向速度 (m/s, 正=远离)

    # ===== 射手2相对目标的特征 (6维) =====
    dist_2: float               # 射手2到目标距离 (m)
    aspect_2: float             # 射手2的姿态角
    closure_rate_2: float       # 射手2的接近率
    off_boresight_2: float      # 射手2的离轴角
    target_v_lateral_2: float   # 目标相对射手2的横向逃逸速度 (m/s) - 关键!
    target_v_radial_2: float    # 目标相对射手2的径向速度 (m/s, 正=远离)

    # ===== 双机协同特征 (3维) =====
    angle_diff: float           # 双机攻击角度差 (度, 越大越好, 90度最佳)
    dist_ratio: float           # 距离比 (两机距离差异)
    time_diff: float            # 导弹到达时间差估计 (s)

    # ===== 目标整体特征 (3维) =====
    target_speed: float         # 目标速度 (m/s)
    target_maneuver: float      # 目标机动强度 (0-1)
    target_is_manned: int       # 目标是否有人机 (0/1)

    # ===== 环境特征 (1维) =====
    avg_altitude: float         # 平均高度 (m)

    def to_array(self) -> np.ndarray:
        """转换为numpy数组用于分类器输入"""
        return np.array([
            # 射手1特征 (6维)
            self.dist_1 / 35000,
            self.aspect_1 / 180,
            self.closure_rate_1 / 800,
            self.off_boresight_1 / 90,
            np.clip(self.target_v_lateral_1 / 400, 0, 1),   # 横向逃逸速度 (关键!)
            np.clip(self.target_v_radial_1 / 400, -1, 1),   # 径向速度

            # 射手2特征 (6维)
            self.dist_2 / 35000,
            self.aspect_2 / 180,
            self.closure_rate_2 / 800,
            self.off_boresight_2 / 90,
            np.clip(self.target_v_lateral_2 / 400, 0, 1),   # 横向逃逸速度 (关键!)
            np.clip(self.target_v_radial_2 / 400, -1, 1),   # 径向速度

            # 协同特征 (3维)
            self.angle_diff / 180,
            self.dist_ratio,
            self.time_diff / 30,

            # 目标整体特征 (3维)
            self.target_speed / 500,
            self.target_maneuver,
            self.target_is_manned,

            # 环境特征 (1维)
            self.avg_altitude / 10000
        ], dtype=np.float32)

    @staticmethod
    def feature_names() -> List[str]:
        """特征名称列表"""
        return [
            # 射手1
            'dist_1', 'aspect_1', 'closure_rate_1', 'off_boresight_1',
            'target_v_lateral_1', 'target_v_radial_1',
            # 射手2
            'dist_2', 'aspect_2', 'closure_rate_2', 'off_boresight_2',
            'target_v_lateral_2', 'target_v_radial_2',
            # 协同
            'angle_diff', 'dist_ratio', 'time_diff',
            # 目标整体
            'target_speed', 'target_maneuver', 'target_is_manned',
            # 环境
            'avg_altitude'
        ]

    @staticmethod
    def feature_dim() -> int:
        """特征维度"""
        return 19


class DualFireFeatureExtractor:
    """
    双机协同开火特征提取器

    从态势数据中提取用于分类器的特征
    """

    # 导弹参数 (基于仿真数据)
    MISSILE_SPEED = 1200  # m/s
    MISSILE_FLIGHT_TIME = 35  # 秒

    def __init__(self):
        pass

    def extract(self, shooter1: Dict, shooter2: Dict, target: Dict) -> DualFireFeatures:
        """
        从两架射手和一个目标提取特征

        Args:
            shooter1: 射手1信息 (platform_list中的元素)
            shooter2: 射手2信息
            target: 目标信息 (track_list中的元素)

        Returns:
            DualFireFeatures: 提取的特征
        """
        # ===== 提取射手1的位置和速度 =====
        s1_lon, s1_lat, s1_alt = self._get_position(shooter1)
        s1_vx, s1_vy, s1_vz = self._get_velocity(shooter1, is_platform=True)
        s1_heading = shooter1.get('heading', 0)

        # ===== 提取射手2的位置和速度 =====
        s2_lon, s2_lat, s2_alt = self._get_position(shooter2)
        s2_vx, s2_vy, s2_vz = self._get_velocity(shooter2, is_platform=True)
        s2_heading = shooter2.get('heading', 0)

        # ===== 提取目标的位置和速度 =====
        t_lon, t_lat, t_alt = self._get_position(target, is_target=True)
        t_vx, t_vy, t_vz = self._get_velocity(target, is_platform=False)
        t_heading = target.get('heading', 0)
        t_roll = target.get('roll', 0)
        t_speed = target.get('speed', 300)
        is_manned = 1 if target.get('platform_entity_type') == '有人机' else 0

        # ===== 计算射手1相对目标的特征 =====
        dist_1 = self._calc_3d_distance(s1_lon, s1_lat, s1_alt, t_lon, t_lat, t_alt)
        aspect_1 = self._calc_aspect_angle(s1_lon, s1_lat, s1_heading, t_lon, t_lat, t_heading)
        closure_rate_1 = self._calc_closure_rate(
            s1_lon, s1_lat, s1_alt, s1_vx, s1_vy, s1_vz,
            t_lon, t_lat, t_alt, t_vx, t_vy, t_vz
        )
        off_boresight_1 = self._calc_off_boresight(s1_lon, s1_lat, s1_heading, t_lon, t_lat)

        # 目标相对于射手1的横向和径向速度 (关键!)
        target_v_lateral_1, target_v_radial_1 = self._calc_target_velocity_decomposition(
            s1_lon, s1_lat, s1_alt, t_lon, t_lat, t_alt, t_vx, t_vy, t_vz
        )

        # ===== 计算射手2相对目标的特征 =====
        dist_2 = self._calc_3d_distance(s2_lon, s2_lat, s2_alt, t_lon, t_lat, t_alt)
        aspect_2 = self._calc_aspect_angle(s2_lon, s2_lat, s2_heading, t_lon, t_lat, t_heading)
        closure_rate_2 = self._calc_closure_rate(
            s2_lon, s2_lat, s2_alt, s2_vx, s2_vy, s2_vz,
            t_lon, t_lat, t_alt, t_vx, t_vy, t_vz
        )
        off_boresight_2 = self._calc_off_boresight(s2_lon, s2_lat, s2_heading, t_lon, t_lat)

        # 目标相对于射手2的横向和径向速度 (关键!)
        target_v_lateral_2, target_v_radial_2 = self._calc_target_velocity_decomposition(
            s2_lon, s2_lat, s2_alt, t_lon, t_lat, t_alt, t_vx, t_vy, t_vz
        )

        # ===== 计算双机协同特征 (核心!) =====
        # 1. 攻击角度差: 两架飞机从目标视角看的方位角之差
        bearing_1 = self._calc_bearing_from_target(t_lon, t_lat, s1_lon, s1_lat)
        bearing_2 = self._calc_bearing_from_target(t_lon, t_lat, s2_lon, s2_lat)
        angle_diff = abs(bearing_1 - bearing_2)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        # 2. 距离比: 表示两机距离的差异
        dist_ratio = min(dist_1, dist_2) / max(dist_1, dist_2) if max(dist_1, dist_2) > 0 else 1.0

        # 3. 导弹到达时间差估计
        tof_1 = dist_1 / self.MISSILE_SPEED  # Time of flight
        tof_2 = dist_2 / self.MISSILE_SPEED
        time_diff = abs(tof_1 - tof_2)

        # ===== 目标整体特征 =====
        # 目标机动强度 (根据roll角)
        target_maneuver = min(abs(t_roll) / 1.4, 1.0)

        # ===== 环境特征 =====
        avg_altitude = (s1_alt + s2_alt + t_alt) / 3

        return DualFireFeatures(
            # 射手1
            dist_1=dist_1,
            aspect_1=aspect_1,
            closure_rate_1=closure_rate_1,
            off_boresight_1=off_boresight_1,
            target_v_lateral_1=target_v_lateral_1,
            target_v_radial_1=target_v_radial_1,
            # 射手2
            dist_2=dist_2,
            aspect_2=aspect_2,
            closure_rate_2=closure_rate_2,
            off_boresight_2=off_boresight_2,
            target_v_lateral_2=target_v_lateral_2,
            target_v_radial_2=target_v_radial_2,
            # 协同
            angle_diff=angle_diff,
            dist_ratio=dist_ratio,
            time_diff=time_diff,
            # 目标整体
            target_speed=t_speed,
            target_maneuver=target_maneuver,
            target_is_manned=is_manned,
            # 环境
            avg_altitude=avg_altitude
        )

    def _calc_target_velocity_decomposition(self, s_lon: float, s_lat: float, s_alt: float,
                                             t_lon: float, t_lat: float, t_alt: float,
                                             t_vx: float, t_vy: float, t_vz: float) -> Tuple[float, float]:
        """
        计算目标相对于射手的速度分解

        这是核心函数! 目标相对于不同射手的横向/径向速度是不同的。

        Args:
            s_lon, s_lat, s_alt: 射手位置
            t_lon, t_lat, t_alt: 目标位置
            t_vx, t_vy, t_vz: 目标速度分量 (东/北/上)

        Returns:
            (v_lateral, v_radial):
            - v_lateral: 目标横向逃逸速度 (垂直于射手->目标连线, 越大越难打)
            - v_radial: 目标径向速度 (沿射手->目标连线, 正=目标远离射手)
        """
        # 计算射手到目标的方向向量 (米)
        dx_m = (t_lon - s_lon) * 111000 * math.cos(math.radians((s_lat + t_lat) / 2))
        dy_m = (t_lat - s_lat) * 111000
        dz_m = t_alt - s_alt

        dist = math.sqrt(dx_m**2 + dy_m**2 + dz_m**2)

        if dist < 1:  # 避免除零
            return 0, 0

        # 单位方向向量 (从射手指向目标，即导弹飞行方向)
        ux, uy, uz = dx_m / dist, dy_m / dist, dz_m / dist

        # 目标速度在导弹来袭方向(反方向)的投影
        # v_radial: 正值表示目标在远离射手 (不利于射手)
        v_radial = t_vx * ux + t_vy * uy + t_vz * uz

        # 目标速度的横向分量 (垂直于连线方向)
        # 这是目标的逃逸能力 - 横向速度越大，导弹越难追上
        t_speed = math.sqrt(t_vx**2 + t_vy**2 + t_vz**2)
        v_lateral_sq = t_speed**2 - v_radial**2
        v_lateral = math.sqrt(max(0, v_lateral_sq))  # 确保非负

        return v_lateral, v_radial

    def _get_position(self, entity: Dict, is_target: bool = False) -> Tuple[float, float, float]:
        """获取实体位置"""
        if is_target:
            # track_list中的目标可能用不同的键名
            lon = entity.get('longitude', entity.get('X', 0))
            lat = entity.get('latitude', entity.get('Y', 0))
            alt = entity.get('altitude', entity.get('Alt', 5000))
        else:
            lon = entity.get('longitude', 0)
            lat = entity.get('latitude', 0)
            alt = entity.get('altitude', 5000)
        return lon, lat, alt

    def _get_velocity(self, entity: Dict, is_platform: bool = True) -> Tuple[float, float, float]:
        """获取实体速度分量"""
        if is_platform:
            # platform_list用 velocity_x/y/z
            vx = entity.get('velocity_x', 0)
            vy = entity.get('velocity_y', 0)
            vz = entity.get('velocity_z', 0)
        else:
            # track_list用 v_x/v_y/v_z
            vx = entity.get('v_x', entity.get('velocity_x', 0))
            vy = entity.get('v_y', entity.get('velocity_y', 0))
            vz = entity.get('v_z', entity.get('velocity_z', 0))
        return vx, vy, vz

    def _calc_3d_distance(self, lon1, lat1, alt1, lon2, lat2, alt2) -> float:
        """计算3D距离 (米)"""
        # 水平距离
        R = 6371000  # 地球半径
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        h_dist = R * c

        # 3D距离
        alt_diff = alt2 - alt1
        return math.sqrt(h_dist**2 + alt_diff**2)

    def _calc_aspect_angle(self, s_lon, s_lat, s_heading, t_lon, t_lat, t_heading) -> float:
        """
        计算姿态角 (从目标视角看射手的方位)
        0 = 尾追 (射手在目标后方)
        180 = 迎头 (射手在目标前方)
        """
        dx = s_lon - t_lon
        dy = s_lat - t_lat
        bearing_to_shooter = math.degrees(math.atan2(dx, dy)) % 360

        target_hdg = math.degrees(t_heading) % 360 if isinstance(t_heading, float) else t_heading % 360
        aspect = abs((bearing_to_shooter - target_hdg + 180) % 360 - 180)
        return aspect

    def _calc_closure_rate(self, s_lon, s_lat, s_alt, s_vx, s_vy, s_vz,
                           t_lon, t_lat, t_alt, t_vx, t_vy, t_vz) -> float:
        """利用速度分量精确计算接近率"""
        # 射手到目标的方向向量
        dx_m = (t_lon - s_lon) * 111000 * math.cos(math.radians((s_lat + t_lat) / 2))
        dy_m = (t_lat - s_lat) * 111000
        dz_m = t_alt - s_alt
        dist = math.sqrt(dx_m**2 + dy_m**2 + dz_m**2)

        if dist < 1:
            return 0

        # 单位方向向量
        ux, uy, uz = dx_m / dist, dy_m / dist, dz_m / dist

        # 相对速度
        rel_vx = t_vx - s_vx
        rel_vy = t_vy - s_vy
        rel_vz = t_vz - s_vz

        # 接近率 (正值表示双方在接近)
        closure_rate = -(rel_vx * ux + rel_vy * uy + rel_vz * uz)
        return closure_rate

    def _calc_off_boresight(self, s_lon, s_lat, s_heading, t_lon, t_lat) -> float:
        """计算离轴角 (射手机头方向与目标方向的夹角)"""
        dx = t_lon - s_lon
        dy = t_lat - s_lat
        bearing_to_target = math.degrees(math.atan2(dx, dy)) % 360

        shooter_hdg = math.degrees(s_heading) % 360 if isinstance(s_heading, float) else s_heading % 360
        off_boresight = abs((bearing_to_target - shooter_hdg + 180) % 360 - 180)
        return off_boresight

    def _calc_bearing_from_target(self, t_lon, t_lat, s_lon, s_lat) -> float:
        """从目标位置计算射手的方位角 (用于计算双机角度差)"""
        dx = s_lon - t_lon
        dy = s_lat - t_lat
        bearing = math.degrees(math.atan2(dx, dy)) % 360
        return bearing


# ==================== 测试代码 ====================

if __name__ == "__main__":
    # 测试特征提取
    extractor = DualFireFeatureExtractor()

    # 模拟两架射手 (不同位置)
    shooter1 = {
        'longitude': 145.9,
        'latitude': 33.5,
        'altitude': 5000,
        'speed': 400,
        'heading': 0.5,
        'velocity_x': 200,
        'velocity_y': 346,
        'velocity_z': 0
    }

    shooter2 = {
        'longitude': 146.1,
        'latitude': 33.3,
        'altitude': 5500,
        'speed': 400,
        'heading': 2.5,
        'velocity_x': -200,
        'velocity_y': 346,
        'velocity_z': 0
    }

    # 模拟目标 (在两架射手之间)
    target = {
        'longitude': 146.0,
        'latitude': 33.4,
        'altitude': 5200,
        'speed': 350,
        'heading': 1.0,
        'v_x': 100,
        'v_y': -300,
        'v_z': 0,
        'roll': 0.5,
        'platform_entity_type': '无人机'
    }

    # 提取特征
    features = extractor.extract(shooter1, shooter2, target)

    print("=" * 60)
    print("双机协同开火特征提取测试")
    print("=" * 60)

    print(f"\n【射手1视角】")
    print(f"  距离: {features.dist_1:.0f}m")
    print(f"  姿态角: {features.aspect_1:.1f}°")
    print(f"  目标横向逃逸速度: {features.target_v_lateral_1:.1f}m/s")
    print(f"  目标径向速度: {features.target_v_radial_1:.1f}m/s (正=远离)")

    print(f"\n【射手2视角】")
    print(f"  距离: {features.dist_2:.0f}m")
    print(f"  姿态角: {features.aspect_2:.1f}°")
    print(f"  目标横向逃逸速度: {features.target_v_lateral_2:.1f}m/s")
    print(f"  目标径向速度: {features.target_v_radial_2:.1f}m/s (正=远离)")

    print(f"\n【核心协同特征】")
    print(f"  攻击角度差: {features.angle_diff:.1f}° (越接近90°越好)")
    print(f"  距离比: {features.dist_ratio:.2f}")
    print(f"  到达时间差: {features.time_diff:.1f}s")

    print(f"\n【物理意义】")
    if abs(features.target_v_lateral_1 - features.target_v_lateral_2) > 50:
        print(f"  ✓ 两射手看到的横向速度差异大 ({abs(features.target_v_lateral_1 - features.target_v_lateral_2):.0f}m/s)")
        print(f"    -> 说明目标难以同时躲避两发导弹!")
    else:
        print(f"  ✗ 两射手看到的横向速度相近")
        print(f"    -> 目标可能同时躲避两发导弹")

    print(f"\n特征向量 ({features.feature_dim()}维):")
    print(features.to_array())
