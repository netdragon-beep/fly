import math
from typing import Tuple


class YxGeoUtils:
    """地理计算工具类"""

    @staticmethod
    def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        """使用Haversine公式计算两点之间的距离(米)"""
        R = 6371000  # 地球半径(米)
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    @staticmethod
    def km_to_lon_lat(battlefield_center_lat: float, distance_km: float, angle_deg: float) -> Tuple[float, float]:
        """将距离和方向转换为经纬度偏移"""
        angle_rad = math.radians(angle_deg)

        # 计算南北方向（纬度）偏移
        lat_km_per_degree = 111.0
        lat_offset = (distance_km * math.sin(angle_rad)) / lat_km_per_degree

        # 计算东西方向（经度）偏移
        lon_km_per_degree = 111.0 * math.cos(math.radians(battlefield_center_lat))  # 考虑纬度修正
        lon_offset = (distance_km * math.cos(angle_rad)) / lon_km_per_degree

        return lon_offset, lat_offset

    @staticmethod
    def calculate_direction_to(from_lon: float, from_lat: float, to_lon: float, to_lat: float) -> float:
        """计算从起点到终点的方向（度）"""
        delta_lon = to_lon - from_lon
        delta_lat = to_lat - from_lat

        angle_rad = math.atan2(delta_lat, delta_lon)
        angle_deg = math.degrees(angle_rad)

        return (angle_deg + 360) % 360

    @staticmethod
    def calculate_bearing(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        """计算从点1到点2的方位角（度）"""
        lon1_rad = math.radians(lon1)
        lat1_rad = math.radians(lat1)
        lon2_rad = math.radians(lon2)
        lat2_rad = math.radians(lat2)

        delta_lon = lon2_rad - lon1_rad

        y = math.sin(delta_lon) * math.cos(lat2_rad)
        x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)

        bearing_rad = math.atan2(y, x)
        bearing_deg = math.degrees(bearing_rad)
        return (bearing_deg + 360) % 360  # 确保在0-360度范围内