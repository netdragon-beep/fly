"""
攻击逻辑动作节点

包含智能火控和Shoot-Look-Shoot策略
"""

from ..bt_framework import Action, NodeStatus
from ..fire_control import SmartFireControl
from utilities.yxScriptTreeFunc import YxScriptTreeFunc as decCmd
from utilities.yxGeoUtils import YxGeoUtils


class ActionAttackLogic(Action):
    """
    攻击逻辑：包含智能火控和机动

    采用 Shoot-Look-Shoot (发射-观察-再发射) 策略：
    - 对同一目标发射导弹后，等待导弹到达（命中或脱靶）
    - 只有确认结果后才考虑发射第二颗导弹
    - 避免浪费弹药的齐射行为
    """

    # 调试开关
    DEBUG_ENABLED = True
    DEBUG_INTERVAL = 25  # 每N帧输出一次火控信息

    # Shoot-Look-Shoot 参数
    MISSILE_FLIGHT_TIME_ESTIMATE = 80  # 估计导弹飞行时间（帧），约8秒@10fps
    MIN_REFIRE_INTERVAL = 40           # 最小再次发射间隔（帧），约4秒

    def tick(self, agent) -> str:
        if not agent.enemy_units:
            return NodeStatus.SUCCESS

        # === 初始化 Shoot-Look-Shoot 追踪器 ===
        if not hasattr(agent, 'pending_missiles'):
            agent.pending_missiles = {}  # {(shooter_name, target_id): launch_frame}

        # === 清理已过期的待定导弹记录 ===
        # 如果发射时间已超过预计飞行时间，认为导弹已到达（命中或脱靶）
        expired_keys = []
        for key, launch_frame in agent.pending_missiles.items():
            if agent.frame_count - launch_frame > self.MISSILE_FLIGHT_TIME_ESTIMATE:
                expired_keys.append(key)
        for key in expired_keys:
            del agent.pending_missiles[key]
            if self.DEBUG_ENABLED:
                print(f"[Shoot-Look-Shoot] 导弹追踪过期: {key[0]} -> 目标{key[1]}")

        # === 检查己方导弹状态（如果有的话）===
        # 通过检测目标是否还存在来判断导弹是否命中
        current_enemy_ids = set()
        for enemy in agent.enemy_units:
            target_id = enemy.get('target_id', enemy.get('id', id(enemy)))
            current_enemy_ids.add(target_id)

        # 如果目标已不存在（被击落），清除相关的pending记录
        keys_to_remove = []
        for key in agent.pending_missiles.keys():
            shooter_name, target_id = key
            if target_id not in current_enemy_ids:
                keys_to_remove.append(key)
                if self.DEBUG_ENABLED:
                    print(f"[Shoot-Look-Shoot] 目标{target_id}已被击落，{shooter_name}可再次开火")
        for key in keys_to_remove:
            del agent.pending_missiles[key]

        # 调试帧计数
        if not hasattr(agent, '_fire_control_debug_frame'):
            agent._fire_control_debug_frame = 0
        should_debug = (agent.frame_count - agent._fire_control_debug_frame >= self.DEBUG_INTERVAL)
        if should_debug:
            agent._fire_control_debug_frame = agent.frame_count

        # Sub-step 1: 智能火控 (Smart Fire Control)
        # 收集所有可能的射击方案，使用SmartFireControl评估
        fire_candidates = []
        for unit in agent.own_units:
            # 检查是否有弹药
            has_ammo = False
            for weapon in unit.get('weapons', []):
                if weapon.get('quantity', 0) > 0:
                    has_ammo = True
                    break
            if not has_ammo:
                continue

            for enemy in agent.enemy_units:
                # 安全获取坐标
                u_lon = unit.get('longitude', unit.get('X', 0))
                u_lat = unit.get('latitude', unit.get('Y', 0))
                e_lon = enemy.get('longitude', enemy.get('X', 0))
                e_lat = enemy.get('latitude', enemy.get('Y', 0))

                if u_lon is None or u_lat is None or e_lon is None or e_lat is None:
                    continue

                # 使用智能火控系统评估
                should_fire, pk, reason = SmartFireControl.should_fire(unit, enemy, agent)

                # 计算距离用于排序
                d = YxGeoUtils.haversine_distance(u_lon, u_lat, e_lon, e_lat)

                # 判断敌机类型：有人机优先级更高
                enemy_type = enemy.get('platform_entity_type', '无人机')
                priority = 0 if enemy_type == '有人机' else 1  # 0=高优先级

                fire_candidates.append({
                    'unit': unit,
                    'enemy': enemy,
                    'dist': d,
                    'priority': priority,
                    'should_fire': should_fire,
                    'pk': pk,
                    'reason': reason
                })

        # 按优先级和Pk排序（优先高价值目标，其次高Pk）
        fire_candidates.sort(key=lambda x: (x['priority'], -x['pk']))

        cur_round_shots = {}  # 本轮发射记录
        fired_units = set()   # 本轮已开火的单位

        # 调试输出
        if self.DEBUG_ENABLED and should_debug and fire_candidates:
            print(f"\n[智能火控] Frame {agent.frame_count}: 评估 {len(fire_candidates)} 个射击方案")
            print(f"  待定导弹数: {len(agent.pending_missiles)}")
            # 输出前5个最佳方案
            for i, cand in enumerate(fire_candidates[:5]):
                unit_name = cand['unit']['name']
                enemy_name = cand['enemy'].get('target_name', cand['enemy'].get('name', '?'))
                print(f"  方案{i+1}: {unit_name} -> {enemy_name}, "
                      f"Pk={cand['pk']:.2f}, 距离={cand['dist']:.0f}m, "
                      f"决策={cand['should_fire']}, 原因={cand['reason']}")

        for cand in fire_candidates:
            u, e = cand['unit'], cand['enemy']

            # 每个单位每帧只开火一次
            if u['name'] in fired_units:
                continue

            # 智能火控判断不应该开火
            if not cand['should_fire']:
                continue

            # 安全获取 target_id
            target_id = e.get('target_id', e.get('id', id(e)))

            # === Shoot-Look-Shoot 检查 ===
            # 检查该射手是否已有导弹正在飞向该目标
            pending_key = (u['name'], target_id)
            if pending_key in agent.pending_missiles:
                launch_frame = agent.pending_missiles[pending_key]
                frames_elapsed = agent.frame_count - launch_frame
                # 如果还没到最小再发射间隔，跳过
                if frames_elapsed < self.MIN_REFIRE_INTERVAL:
                    if self.DEBUG_ENABLED and should_debug:
                        print(f"  [Shoot-Look-Shoot] {u['name']} 等待上一枚导弹结果 "
                              f"(已过{frames_elapsed}帧/{self.MISSILE_FLIGHT_TIME_ESTIMATE}帧)")
                    continue

            # 敌机未被过度攻击 (最多4发导弹)
            already_fired = e.get('is_fired_num', 0) + cur_round_shots.get(target_id, 0)
            if already_fired >= 4:
                continue

            # 安全获取 target_name
            target_name = e.get('target_name', e.get('name', ''))
            if not target_name:
                continue

            # 发射！
            if agent.try_fire_weapon(u):
                agent.add_action(decCmd.fire_track(u['name'], target_name), None)
                cur_round_shots[target_id] = cur_round_shots.get(target_id, 0) + 1
                fired_units.add(u['name'])

                # === 记录 Shoot-Look-Shoot 追踪 ===
                agent.pending_missiles[pending_key] = agent.frame_count

                if self.DEBUG_ENABLED:
                    print(f"[开火] {u['name']} -> {target_name}, Pk={cand['pk']:.2f}, "
                          f"距离={cand['dist']:.0f}m, 原因={cand['reason']}")

        # Sub-step 2: Maneuver to Attack (仅对未被占用的单位有效)
        for unit in agent.own_units:
            if unit['name'] in agent.commanded_units:
                continue  # 正在规避的单位不执行进攻机动

            u_lon = unit.get('longitude', unit.get('X'))
            u_lat = unit.get('latitude', unit.get('Y'))
            if u_lon is None or u_lat is None:
                continue

            # 找最近敌机（优先有人机）
            closest = None
            min_d = float('inf')
            best_priority = 999

            for enemy in agent.enemy_units:
                e_lon = enemy.get('longitude', enemy.get('X'))
                e_lat = enemy.get('latitude', enemy.get('Y'))
                if e_lon is None or e_lat is None:
                    continue

                d = YxGeoUtils.haversine_distance(u_lon, u_lat, e_lon, e_lat)
                enemy_type = enemy.get('platform_entity_type', '无人机')
                priority = 0 if enemy_type == '有人机' else 1

                # 优先级更高 或 同优先级但更近
                if priority < best_priority or (priority == best_priority and d < min_d):
                    min_d = d
                    closest = enemy
                    best_priority = priority

            if closest:
                e_lon = closest.get('longitude', closest.get('X', 0))
                e_lat = closest.get('latitude', closest.get('Y', 0))
                e_alt = closest.get('altitude', closest.get('Alt', 3000))

                # 使用智能火控的最优攻击距离
                is_manned = unit.get('type') == '有人机'
                optimal_dist = SmartFireControl.MANNED_OPTIMAL_RANGE if is_manned else SmartFireControl.UAV_OPTIMAL_RANGE

                # 如果已经在最优距离内，不需要继续接近
                if min_d <= optimal_dist:
                    # 维持当前位置或轻微调整
                    continue

                # 飞向攻击占位点（最优射程位置）
                direction = YxGeoUtils.calculate_direction_to(e_lon, e_lat, u_lon, u_lat)
                lon_off, lat_off = YxGeoUtils.km_to_lon_lat(agent.center_lat, optimal_dist / 1000, direction)

                target_pt = (e_lat + lat_off, e_lon + lon_off, e_alt)
                # 速度根据距离调整：远的快接近，近的慢接近
                approach_speed = 550 if min_d > optimal_dist * 1.5 else 450
                agent.add_action(decCmd.fly_to_point(unit['name'], target_pt, approach_speed), unit['name'])

        return NodeStatus.SUCCESS
