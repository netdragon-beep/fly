"""
导弹规避动作节点

Cranking机动 + 高度规避
"""

import math
from ..bt_framework import Action, NodeStatus
from utilities.yxScriptTreeFunc import YxScriptTreeFunc as decCmd
from utilities.yxGeoUtils import YxGeoUtils


class ActionEvadeMissiles(Action):
    """
    多导弹协同规避 - Cranking机动 + 高度规避

    核心改进（解决导弹追踪问题）：
    1. Cranking机动：不是垂直逃跑，而是斜向后方飞行（约120-150°角度）
       - 既增加横向距离，又增加纵向距离
       - 持续让导弹需要转向追踪，消耗导弹能量
    2. 高度骤降：紧急时快速下降，利用高度差增加导弹追踪难度
    3. 更早规避：35km就开始规避，给自己更多反应时间
    4. 更大规避距离：规避点设置得更远

    导弹参数（假设）：
    - 导弹速度：1200 m/s
    - 飞机最大速度：900 m/s
    - 导弹能量有限，持续转向会消耗能量导致失速
    """
    # === 威胁判断参数 ===
    MISSILE_THREAT_RADIUS = 35000   # 威胁半径35km（更早规避）
    MISSILE_CRITICAL_RADIUS = 12000 # 紧急规避半径12km
    MISSILE_DANGER_RADIUS = 20000   # 危险半径20km
    MISSILE_THREAT_ANGLE = 90       # 导弹威胁角度（收紧到90度，减少误判）

    # === 规避行为参数 ===
    EVADE_SPEED_NORMAL = 800        # 正常规避速度
    EVADE_SPEED_CRITICAL = 900      # 紧急规避速度（最大）
    EVADE_DISTANCE_NORMAL = 20      # 正常规避距离（km）- 增大到20km
    EVADE_DISTANCE_CRITICAL = 25    # 紧急规避距离（km）- 增大到25km

    # === Cranking机动参数 ===
    CRANK_ANGLE_NORMAL = 120        # 正常Crank角度（向后斜飞120°）
    CRANK_ANGLE_CRITICAL = 150      # 紧急Crank角度（更向后，150°）

    # === 高度规避参数 ===
    ALTITUDE_DROP_NORMAL = 800      # 正常下降高度
    ALTITUDE_DROP_CRITICAL = 1500   # 紧急下降高度（大幅下降）
    MIN_ALTITUDE = 500              # 最低高度限制
    MAX_ALTITUDE = 6000             # 最高高度限制

    # === 边界安全 ===
    BOUNDARY_MARGIN = 0.12          # 边界安全边距（约13km）

    # === 调试 ===
    DEBUG_ENABLED = True
    DEBUG_INTERVAL = 20

    def tick(self, agent) -> str:
        # 调试输出
        if self.DEBUG_ENABLED:
            if not hasattr(agent, '_missile_debug_frame'):
                agent._missile_debug_frame = 0
            if agent.frame_count - agent._missile_debug_frame >= self.DEBUG_INTERVAL:
                agent._missile_debug_frame = agent.frame_count
                print(f"[导弹规避] Frame {agent.frame_count}: 检测到 {len(agent.enemy_missiles)} 枚敌方导弹")

        if not agent.enemy_missiles:
            return NodeStatus.SUCCESS

        # 收集导弹信息
        all_missiles = []
        for missile in agent.enemy_missiles:
            m_lon = missile.get('longitude', 0)
            m_lat = missile.get('latitude', 0)
            m_heading = math.degrees(missile.get('heading', 0)) % 360
            m_speed = missile.get('speed', 1200)
            if m_lon and m_lat:
                all_missiles.append({
                    'lon': m_lon, 'lat': m_lat,
                    'heading': m_heading, 'speed': m_speed,
                    'id': missile.get('target_id') or missile.get('name') or id(missile)
                })

        if not all_missiles:
            return NodeStatus.SUCCESS

        # 调试：输出导弹信息
        if self.DEBUG_ENABLED and agent.frame_count - agent._missile_debug_frame < 3:
            for i, m in enumerate(all_missiles):
                print(f"  导弹{i+1}: ({m['lat']:.4f}, {m['lon']:.4f}), 航向={m['heading']:.1f}°")

        evade_count = 0
        for unit in agent.own_units:
            unit_name = unit['name']
            u_lon = unit.get('longitude', 0)
            u_lat = unit.get('latitude', 0)
            u_alt = unit.get('altitude', 3000)

            if not u_lon or not u_lat:
                continue

            # === 分析威胁导弹 ===
            threat_missiles = []
            min_dist = float('inf')
            primary_threat = None  # 最近的威胁导弹

            for missile in all_missiles:
                dist = YxGeoUtils.haversine_distance(u_lon, u_lat, missile['lon'], missile['lat'])

                if dist > self.MISSILE_THREAT_RADIUS:
                    continue

                # 检查是否在导弹飞行方向上
                bearing_to_unit = YxGeoUtils.calculate_bearing(
                    missile['lon'], missile['lat'], u_lon, u_lat
                )
                angle_diff = abs((bearing_to_unit - missile['heading'] + 180) % 360 - 180)

                # 调试输出
                if self.DEBUG_ENABLED and agent.frame_count - agent._missile_debug_frame < 3:
                    print(f"    {unit_name}: 距导弹{dist:.0f}m, 角度差={angle_diff:.1f}°")

                if angle_diff < self.MISSILE_THREAT_ANGLE:
                    threat_missiles.append({
                        **missile,
                        'dist': dist,
                        'bearing_to_unit': bearing_to_unit
                    })
                    if dist < min_dist:
                        min_dist = dist
                        primary_threat = threat_missiles[-1]

            # === 没有威胁，跳过 ===
            if not threat_missiles:
                continue

            # === 确定威胁等级 ===
            is_critical = min_dist < self.MISSILE_CRITICAL_RADIUS
            is_danger = min_dist < self.MISSILE_DANGER_RADIUS

            if is_critical:
                evade_speed = self.EVADE_SPEED_CRITICAL
                evade_dist = self.EVADE_DISTANCE_CRITICAL
                crank_angle = self.CRANK_ANGLE_CRITICAL
                alt_drop = self.ALTITUDE_DROP_CRITICAL
                mode_str = "[紧急]"
            elif is_danger:
                evade_speed = self.EVADE_SPEED_CRITICAL
                evade_dist = self.EVADE_DISTANCE_NORMAL
                crank_angle = self.CRANK_ANGLE_NORMAL
                alt_drop = self.ALTITUDE_DROP_NORMAL
                mode_str = "[危险]"
            else:
                evade_speed = self.EVADE_SPEED_NORMAL
                evade_dist = self.EVADE_DISTANCE_NORMAL
                crank_angle = self.CRANK_ANGLE_NORMAL
                alt_drop = self.ALTITUDE_DROP_NORMAL // 2
                mode_str = "[正常]"

            # === 计算 Cranking 规避方向 ===
            # Cranking: 向导弹来向的斜后方飞行
            # 导弹航向 -> 我要飞向导弹的反方向再偏转crank_angle
            missile_heading = primary_threat['heading']

            # 反向（背对导弹）
            away_dir = (missile_heading + 180) % 360

            # 选择左Crank还是右Crank（选择离边界远的方向）
            left_crank = (away_dir - (180 - crank_angle)) % 360
            right_crank = (away_dir + (180 - crank_angle)) % 360

            # 计算两个方向的目标点
            left_lon_off, left_lat_off = YxGeoUtils.km_to_lon_lat(u_lat, evade_dist, left_crank)
            right_lon_off, right_lat_off = YxGeoUtils.km_to_lon_lat(u_lat, evade_dist, right_crank)

            left_target_lon = u_lon + left_lon_off
            left_target_lat = u_lat + left_lat_off
            right_target_lon = u_lon + right_lon_off
            right_target_lat = u_lat + right_lat_off

            # 检查边界安全性
            left_safe = self._check_boundary_safe(left_target_lon, left_target_lat, agent)
            right_safe = self._check_boundary_safe(right_target_lon, right_target_lat, agent)

            # 选择规避方向
            if left_safe and not right_safe:
                evade_dir = left_crank
            elif right_safe and not left_safe:
                evade_dir = right_crank
            else:
                # 两边都安全或都不安全，选择离战场中心更近的
                left_dist_center = abs(left_target_lat - agent.center_lat) + abs(left_target_lon - agent.center_lon)
                right_dist_center = abs(right_target_lat - agent.center_lat) + abs(right_target_lon - agent.center_lon)
                evade_dir = left_crank if left_dist_center < right_dist_center else right_crank

            # === 计算规避目标点 ===
            lon_off, lat_off = YxGeoUtils.km_to_lon_lat(u_lat, evade_dist, evade_dir)
            evade_lon = u_lon + lon_off
            evade_lat = u_lat + lat_off

            # === 高度规避：紧急时下降 ===
            # 下降可以增加导弹的俯角追踪难度，同时利用地面杂波
            if is_critical:
                # 紧急下降
                evade_alt = max(u_alt - alt_drop, self.MIN_ALTITUDE)
            elif is_danger:
                # 交替上升/下降
                if unit_name.endswith(('1', '3')):
                    evade_alt = max(u_alt - alt_drop, self.MIN_ALTITUDE)
                else:
                    evade_alt = min(u_alt + alt_drop // 2, self.MAX_ALTITUDE)
            else:
                evade_alt = u_alt  # 保持高度

            # === 执行规避 ===
            agent.add_action(
                decCmd.fly_to_point(unit_name, (evade_lat, evade_lon, evade_alt), evade_speed),
                unit_name
            )
            evade_count += 1

            # 调试输出
            if self.DEBUG_ENABLED:
                if is_critical:
                    print(f"    [紧急] {unit_name} 距离={min_dist:.0f}m")
                print(f"[规避执行] {mode_str} {unit_name} 规避 {len(threat_missiles)} 枚导弹，"
                      f"最近距离={min_dist:.0f}m, Crank={crank_angle}°, "
                      f"目标=({evade_lat:.4f}, {evade_lon:.4f}, {evade_alt:.0f}m)")

        if self.DEBUG_ENABLED and evade_count > 0:
            print(f"[导弹规避] Frame {agent.frame_count}: {evade_count} 个单位执行Cranking规避")

        return NodeStatus.SUCCESS

    def _check_boundary_safe(self, lon, lat, agent):
        """检查坐标是否在安全边界内"""
        bf = agent.battlefield
        if bf.get('min_lon') is None:
            return True
        margin = self.BOUNDARY_MARGIN
        return (bf['min_lon'] + margin < lon < bf['max_lon'] - margin and
                bf['min_lat'] + margin < lat < bf['max_lat'] - margin)
