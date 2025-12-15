"""
攻击逻辑动作节点

包含智能火控和Shoot-Look-Shoot策略
"""

from typing import Dict
from ..bt_framework import Action, NodeStatus
from ..fire_control import SmartFireControl
from utilities.yxScriptTreeFunc import YxScriptTreeFunc as decCmd
from utilities.yxGeoUtils import YxGeoUtils


class ActionAttackLogic(Action):
    """
    攻击逻辑：包含智能火控和机动

    采用 双机协同开火 (Coordinated Fire) 策略：
    - 必须两架飞机同时对同一目标各发射一发导弹
    - 不允许单机独立开火（避免浪费弹药）
    - 双机从不同方向攻击，目标难以同时规避
    - 每次协同攻击消耗2发导弹，但命中率大幅提升
    """

    # 调试开关
    DEBUG_ENABLED = False  # 开启调试便于观察协同开火
    DEBUG_INTERVAL = 10  # 每N帧输出一次火控信息

    # 协同开火参数（基于官方导弹参数：速度1200m/s，最大飞行时间60s）
    # 有效射程35km，导弹飞行约30秒到达
    MISSILE_FLIGHT_TIME_ESTIMATE = 350  # 估计导弹飞行时间（帧），约35秒@10fps（考虑最远有效射程）
    MIN_REFIRE_INTERVAL = 100           # 最小再次发射间隔（帧），约10秒
    MAX_PENDING_MISSILES_PER_TARGET = 2  # 每个目标最多同时有2枚导弹（双机各1发）

    # 双机协同开火配置
    REQUIRE_COORDINATED_FIRE = True    # 是否强制要求双机协同（True=必须双机同时开火）
    COORDINATED_FIRE_WINDOW = 3        # 协同开火时间窗口（帧），两机必须在此窗口内都能开火
    MIN_ANGLE_DIFFERENCE = 30          # 最小攻击角度差（度），确保从不同方向攻击

    # 数据收集开关（用于拟合致死区间模型）
    COLLECT_KILL_DATA = True

    def tick(self, agent) -> str:
        if not agent.enemy_units:
            return NodeStatus.SUCCESS

        # === 初始化 Shoot-Look-Shoot 追踪器 ===
        if not hasattr(agent, 'pending_missiles'):
            agent.pending_missiles = {}  # {(shooter_name, target_id): launch_frame}

        # === 初始化导弹参数数据收集器（用于拟合致死区间模型）===
        if not hasattr(agent, 'missile_launch_data'):
            agent.missile_launch_data = {}  # {(shooter_name, target_id): {launch_params...}}
        if not hasattr(agent, 'kill_data_records'):
            agent.kill_data_records = []  # [{...params, result: 'kill'/'miss'/'timeout'}]

        # === 清理已过期的待定导弹记录 ===
        # 如果发射时间已超过预计飞行时间，认为导弹已到达（命中或脱靶）
        expired_keys = []
        for key, launch_frame in agent.pending_missiles.items():
            if agent.frame_count - launch_frame > self.MISSILE_FLIGHT_TIME_ESTIMATE:
                expired_keys.append(key)
        for key in expired_keys:
            del agent.pending_missiles[key]
            if self.DEBUG_ENABLED:
                print(f"[导弹超时] {key[0]}的导弹飞行超时(可能脱靶) -> 目标{key[1]}")

            # === 数据收集：记录超时（脱靶）===
            if self.COLLECT_KILL_DATA and key in agent.missile_launch_data:
                record = agent.missile_launch_data.pop(key)
                record['result'] = 'timeout'
                record['flight_frames'] = agent.frame_count - record['launch_frame']
                agent.kill_data_records.append(record)
                print(f"[数据收集] 脱靶: dist={record['distance']:.0f}m, "
                      f"aspect={record['aspect_angle']:.1f}°, "
                      f"closure={record['closure_rate']:.1f}m/s, Pk={record['pk']:.2f}")

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
                elapsed = agent.frame_count - agent.pending_missiles[key]
                if self.DEBUG_ENABLED:
                    print(f"\n{'*'*40}")
                    print(f"[命中!!!] {shooter_name}的导弹命中目标!")
                    print(f"  目标ID: {target_id}")
                    print(f"  飞行时间: {elapsed}帧")
                    print(f"{'*'*40}")

                # === 数据收集：记录击杀 ===
                if self.COLLECT_KILL_DATA and key in agent.missile_launch_data:
                    record = agent.missile_launch_data.pop(key)
                    record['result'] = 'kill'
                    record['flight_frames'] = elapsed
                    # 估算导弹实际飞行距离和命中时的相对距离
                    flight_time_sec = elapsed * 0.1  # 假设10fps
                    missile_speed = record.get('missile_speed', 800)
                    combined_speed = record.get('combined_speed', missile_speed)
                    record['est_flight_distance'] = flight_time_sec * missile_speed
                    record['est_intercept_distance'] = record['distance'] - flight_time_sec * combined_speed
                    agent.kill_data_records.append(record)
                    # print(f"[数据收集] 击杀! dist={record['distance']:.0f}m, "
                    #       f"aspect={record['aspect_angle']:.1f}°, "
                    #       f"closure={record['closure_rate']:.1f}m/s, Pk={record['pk']:.2f}, "
                    #       f"飞行={elapsed}帧, 估算拦截距离={record['est_intercept_distance']:.0f}m")

        for key in keys_to_remove:
            del agent.pending_missiles[key]
            # 清理launch_data中的残留（如果还有的话）
            if key in agent.missile_launch_data:
                del agent.missile_launch_data[key]

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

            # === 有人机安全检查 ===
            # 有人机只在40km内最多1架有弹药敌机时才参与攻击
            if unit.get('type') == '有人机':
                if not self._should_manned_attack(agent, unit):
                    if self.DEBUG_ENABLED and should_debug:
                        print(f"[有人机安全] {unit['name']} 周围有多架有弹药敌机，暂不参与攻击")
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

        # 调试输出 - 详细火控信息
        if self.DEBUG_ENABLED and should_debug:
            print(f"\n{'='*60}")
            print(f"[火控系统-双机协同] Frame {agent.frame_count}")
            print(f"{'='*60}")

            # 己方单位弹药状态
            print(f"[己方单位弹药状态]")
            for unit in agent.own_units:
                ammo_count = 0
                for weapon in unit.get('weapons', []):
                    ammo_count += weapon.get('quantity', 0)
                print(f"  {unit['name']}: 剩余弹药={ammo_count}")

            # 敌方单位状态
            print(f"[敌方单位] 共 {len(agent.enemy_units)} 个")
            for enemy in agent.enemy_units:
                e_name = enemy.get('target_name', enemy.get('name', '?'))
                e_type = enemy.get('platform_entity_type', '?')
                e_fired = enemy.get('is_fired_num', 0)
                print(f"  {e_name} ({e_type}): 已被攻击{e_fired}次")

            # 待定导弹（按目标分组显示）
            print(f"[待定导弹] 共 {len(agent.pending_missiles)} 枚 (每目标限制: {self.MAX_PENDING_MISSILES_PER_TARGET}枚)")
            # 按目标分组
            missiles_by_target = {}
            for key, frame in agent.pending_missiles.items():
                shooter_name, target_id = key
                if target_id not in missiles_by_target:
                    missiles_by_target[target_id] = []
                missiles_by_target[target_id].append((shooter_name, frame))
            for target_id, missiles in missiles_by_target.items():
                print(f"  目标{target_id}: {len(missiles)}枚导弹在途")
                for shooter_name, frame in missiles:
                    elapsed = agent.frame_count - frame
                    print(f"    - {shooter_name}: 已飞行{elapsed}帧")

        # === 双机协同开火逻辑 ===
        if self.REQUIRE_COORDINATED_FIRE:
            # 按目标分组候选射击方案
            candidates_by_target = {}
            for cand in fire_candidates:
                if not cand['should_fire']:
                    continue
                target_id = cand['enemy'].get('target_id', cand['enemy'].get('id', id(cand['enemy'])))
                if target_id not in candidates_by_target:
                    candidates_by_target[target_id] = []
                candidates_by_target[target_id].append(cand)

            if self.DEBUG_ENABLED and should_debug:
                print(f"\n[协同开火分析]")
                for target_id, cands in candidates_by_target.items():
                    target_name = cands[0]['enemy'].get('target_name', '?')
                    shooters = [c['unit']['name'] for c in cands]
                    print(f"  目标 {target_name}: 可用射手 {shooters}")

            # 寻找可以协同开火的目标
            for target_id, cands in candidates_by_target.items():
                # 需要至少2架飞机可以对同一目标开火
                if len(cands) < 2:
                    if self.DEBUG_ENABLED and should_debug:
                        target_name = cands[0]['enemy'].get('target_name', '?')
                        print(f"  [跳过] 目标 {target_name}: 只有1架飞机可开火，需要双机协同")
                    continue

                # 检查该目标是否已有导弹在途
                missiles_to_target = sum(1 for key in agent.pending_missiles if key[1] == target_id)
                if missiles_to_target >= self.MAX_PENDING_MISSILES_PER_TARGET:
                    if self.DEBUG_ENABLED and should_debug:
                        target_name = cands[0]['enemy'].get('target_name', '?')
                        print(f"  [跳过] 目标 {target_name}: 已有{missiles_to_target}枚导弹在途")
                    continue

                # 选择最佳的两架飞机进行协同攻击
                # 使用V2物理NEZ模型评估最佳配对
                best_pair = None
                best_combined_pk = 0
                best_nez_info = None
                best_fire_reason = ""

                for i in range(len(cands)):
                    for j in range(i + 1, len(cands)):
                        unit1 = cands[i]['unit']
                        unit2 = cands[j]['unit']
                        enemy = cands[i]['enemy']

                        # 检查两架飞机是否都还没开火
                        if unit1['name'] in fired_units or unit2['name'] in fired_units:
                            continue

                        # 检查两架飞机是否都有弹药
                        if not self._has_ammo(unit1) or not self._has_ammo(unit2):
                            continue

                        # 检查两架飞机是否都没有对该目标的待定导弹
                        key1 = (unit1['name'], target_id)
                        key2 = (unit2['name'], target_id)
                        if key1 in agent.pending_missiles or key2 in agent.pending_missiles:
                            continue

                        # 使用V2物理NEZ模型评估协同开火
                        should_fire, combined_pk, nez_info, reason = SmartFireControl.should_coordinated_fire(
                            unit1, unit2, enemy, agent
                        )

                        # 选择综合Pk最高的配对
                        if should_fire and combined_pk > best_combined_pk:
                            best_combined_pk = combined_pk
                            best_pair = (cands[i], cands[j])
                            best_nez_info = nez_info
                            best_fire_reason = reason

                # 如果找到合适的配对，执行协同开火
                if best_pair and best_combined_pk >= SmartFireControl.PK_THRESHOLD_COORDINATED:
                    cand1, cand2 = best_pair
                    unit1, unit2 = cand1['unit'], cand2['unit']
                    enemy = cand1['enemy']
                    target_name = enemy.get('target_name', enemy.get('name', ''))

                    if not target_name:
                        continue

                    # 检查敌机未被过度攻击
                    already_fired = enemy.get('is_fired_num', 0) + cur_round_shots.get(target_id, 0)
                    if already_fired >= 4:
                        continue

                    # 双机同时发射！
                    fire_success_1 = agent.try_fire_weapon(unit1)
                    fire_success_2 = agent.try_fire_weapon(unit2)

                    if fire_success_1 and fire_success_2:
                        # 两架都成功发射
                        agent.add_action(decCmd.fire_track(unit1['name'], target_name), None)
                        agent.add_action(decCmd.fire_track(unit2['name'], target_name), None)

                        cur_round_shots[target_id] = cur_round_shots.get(target_id, 0) + 2
                        fired_units.add(unit1['name'])
                        fired_units.add(unit2['name'])

                        # 记录待定导弹
                        pending_key1 = (unit1['name'], target_id)
                        pending_key2 = (unit2['name'], target_id)
                        agent.pending_missiles[pending_key1] = agent.frame_count
                        agent.pending_missiles[pending_key2] = agent.frame_count

                        # # 详细输出协同开火信息
                        # print(f"\n{'*'*60}")
                        # print(f"[双机协同开火!!! - V2物理NEZ]")
                        # print(f"  射手1: {unit1['name']}")
                        # print(f"    距离: {best_nez_info['dist1']/1000:.2f}km, NEZ: {best_nez_info['nez1']/1000:.2f}km")
                        # print(f"    Pk: {best_nez_info['pk1']:.2f}, 径向速度: {best_nez_info['v_radial_1']:.1f}m/s, 横向速度: {best_nez_info['v_lateral_1']:.1f}m/s")
                        # print(f"  射手2: {unit2['name']}")
                        # print(f"    距离: {best_nez_info['dist2']/1000:.2f}km, NEZ: {best_nez_info['nez2']/1000:.2f}km")
                        # print(f"    Pk: {best_nez_info['pk2']:.2f}, 径向速度: {best_nez_info['v_radial_2']:.1f}m/s, 横向速度: {best_nez_info['v_lateral_2']:.1f}m/s")
                        # print(f"  目标: {target_name}")
                        # print(f"  综合Pk: {best_combined_pk:.2f}")
                        # print(f"  逃逸难度: {best_nez_info['escape_difficulty']:.2f}")
                        # print(f"  协同NEZ: {best_nez_info['combined_nez']/1000:.2f}km")
                        # print(f"  开火原因: {best_fire_reason}")
                        # print(f"{'*'*60}")

                        # 收集发射数据（为两架飞机分别记录）
                        self._record_launch_data(agent, unit1, enemy, cand1, pending_key1)
                        self._record_launch_data(agent, unit2, enemy, cand2, pending_key2)

                    elif fire_success_1 or fire_success_2:
                        # 只有一架成功发射 - 这不应该发生，但如果发生了要处理
                        if fire_success_1:
                            agent.add_action(decCmd.fire_track(unit1['name'], target_name), None)
                            fired_units.add(unit1['name'])
                            pending_key1 = (unit1['name'], target_id)
                            agent.pending_missiles[pending_key1] = agent.frame_count
                            # print(f"[警告] 协同开火失败，只有 {unit1['name']} 成功发射")
                        if fire_success_2:
                            agent.add_action(decCmd.fire_track(unit2['name'], target_name), None)
                            fired_units.add(unit2['name'])
                            pending_key2 = (unit2['name'], target_id)
                            agent.pending_missiles[pending_key2] = agent.frame_count
                            # print(f"[警告] 协同开火失败，只有 {unit2['name']} 成功发射")

                elif best_pair:
                    if self.DEBUG_ENABLED and should_debug:
                        target_name = cands[0]['enemy'].get('target_name', '?')
                        print(f"  [跳过] 目标 {target_name}: 综合Pk {best_combined_pk:.2f} < {SmartFireControl.PK_THRESHOLD_COORDINATED}")
        else:
            # 原始单机开火逻辑（作为备用）
            for cand in fire_candidates:
                u, e = cand['unit'], cand['enemy']

                if u['name'] in fired_units:
                    continue

                if not cand['should_fire']:
                    continue

                target_id = e.get('target_id', e.get('id', id(e)))

                missiles_to_target = sum(1 for key in agent.pending_missiles if key[1] == target_id)
                if missiles_to_target >= self.MAX_PENDING_MISSILES_PER_TARGET:
                    continue

                pending_key = (u['name'], target_id)
                if pending_key in agent.pending_missiles:
                    launch_frame = agent.pending_missiles[pending_key]
                    frames_elapsed = agent.frame_count - launch_frame
                    if frames_elapsed < self.MIN_REFIRE_INTERVAL:
                        continue

                already_fired = e.get('is_fired_num', 0) + cur_round_shots.get(target_id, 0)
                if already_fired >= 4:
                    continue

                target_name = e.get('target_name', e.get('name', ''))
                if not target_name:
                    continue

                if agent.try_fire_weapon(u):
                    agent.add_action(decCmd.fire_track(u['name'], target_name), None)
                    cur_round_shots[target_id] = cur_round_shots.get(target_id, 0) + 1
                    fired_units.add(u['name'])

                    # === 记录 Shoot-Look-Shoot 追踪 ===
                    agent.pending_missiles[pending_key] = agent.frame_count

                    # 收集发射数据
                    self._record_launch_data(agent, u, e, cand, pending_key)

                    if self.DEBUG_ENABLED:
                        print(f"\n[单机开火] {u['name']} -> {target_name}")

        # Sub-step 2: Maneuver to Attack (仅对未被占用的单位有效)
        # 使用滞后追踪（Lag Pursuit）策略，飞向敌机后方而非当前位置
        # 这样可以自然进入尾追位置，避免迎头对峙
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

                # 【滞后追踪（Lag Pursuit）】
                # 不再直接飞向敌机当前位置，而是飞向敌机后方
                # 这样可以自然进入尾追位置，避免迎头对峙
                lag_point = self._calculate_lag_pursuit_point(unit, closest, optimal_dist)

                # 速度根据距离调整：远的快接近，近的慢接近
                approach_speed = 500 if is_manned else 340
                if min_d > optimal_dist * 1.5:
                    approach_speed = 550 if is_manned else 360

                agent.add_action(decCmd.fly_to_point(unit['name'], lag_point, approach_speed), unit['name'])

        return NodeStatus.SUCCESS

    @classmethod
    def save_kill_data(cls, agent, filepath=None):
        """
        保存击杀数据到文件（用于后续分析和拟合模型）

        Args:
            agent: 包含kill_data_records的agent实例
            filepath: 保存路径，默认为当前目录下的kill_data_v1.json
        """
        import json
        import os

        if not hasattr(agent, 'kill_data_records') or not agent.kill_data_records:
            print("[数据收集] 没有收集到数据")
            return

        if filepath is None:
            # v1版本数据文件
            filepath = os.path.join(os.path.dirname(__file__), '..', 'kill_data_v1.json')

        # 读取已有数据（如果存在）
        existing_data = []
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except:
                existing_data = []

        # 添加新数据
        existing_data.extend(agent.kill_data_records)

        # 保存
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

        print(f"[数据收集] 保存了 {len(agent.kill_data_records)} 条记录到 {filepath}")
        print(f"[数据收集] 总共 {len(existing_data)} 条记录")

        # 打印统计
        # cls.print_kill_data_summary(agent.kill_data_records)

    @classmethod
    def print_kill_data_summary(cls, records):
        """打印击杀数据统计摘要"""
        if not records:
            return

        kills = [r for r in records if r['result'] == 'kill']
        misses = [r for r in records if r['result'] in ('timeout', 'miss')]

        print(f"\n{'='*60}")
        print(f"[致死区间分析] 共 {len(records)} 条数据")
        print(f"{'='*60}")
        print(f"击杀: {len(kills)} ({100*len(kills)/len(records):.1f}%)")
        print(f"脱靶: {len(misses)} ({100*len(misses)/len(records):.1f}%)")

        if kills:
            avg_kill_dist = sum(r['distance'] for r in kills) / len(kills)
            avg_kill_aspect = sum(r['aspect_angle'] for r in kills) / len(kills)
            avg_kill_closure = sum(r['closure_rate'] for r in kills) / len(kills)
            avg_kill_pk = sum(r['pk'] for r in kills) / len(kills)
            in_nez_kills = sum(1 for r in kills if r['in_nez'])

            print(f"\n[击杀条件统计]")
            print(f"  平均距离: {avg_kill_dist/1000:.2f} km")
            print(f"  平均姿态角: {avg_kill_aspect:.1f}°")
            print(f"  平均接近率: {avg_kill_closure:.1f} m/s")
            print(f"  平均Pk: {avg_kill_pk:.2f}")
            print(f"  在NEZ内: {in_nez_kills}/{len(kills)} ({100*in_nez_kills/len(kills):.1f}%)")

        if misses:
            avg_miss_dist = sum(r['distance'] for r in misses) / len(misses)
            avg_miss_aspect = sum(r['aspect_angle'] for r in misses) / len(misses)
            avg_miss_closure = sum(r['closure_rate'] for r in misses) / len(misses)
            avg_miss_pk = sum(r['pk'] for r in misses) / len(misses)
            in_nez_misses = sum(1 for r in misses if r['in_nez'])

            print(f"\n[脱靶条件统计]")
            print(f"  平均距离: {avg_miss_dist/1000:.2f} km")
            print(f"  平均姿态角: {avg_miss_aspect:.1f}°")
            print(f"  平均接近率: {avg_miss_closure:.1f} m/s")
            print(f"  平均Pk: {avg_miss_pk:.2f}")
            print(f"  在NEZ内: {in_nez_misses}/{len(misses)} ({100*in_nez_misses/len(misses):.1f}%)")

        print(f"{'='*60}\n")

    # 滞后追踪（Lag Pursuit）参数
    LAG_PURSUIT_DISTANCE_KM = 10        # 滞后追踪距离 10km（敌机后方）
    LAG_PURSUIT_MIN_ASPECT = 60         # 最小姿态角才使用滞后追踪（避免已在尾追位置时还绕后）

    def _calculate_lag_pursuit_point(self, unit: Dict, enemy: Dict, optimal_dist: float) -> tuple:
        """
        计算滞后追踪点（Lag Pursuit Point）

        滞后追踪是BFM（Basic Fighter Maneuvers）中的核心概念：
        - 不直接飞向敌机当前位置
        - 而是飞向敌机后方（敌机航向的反方向）
        - 这样可以自然进入尾追位置，避免迎头对峙

        Args:
            unit: 己方单位
            enemy: 敌方单位
            optimal_dist: 最优攻击距离

        Returns:
            (target_lat, target_lon, target_alt)
        """
        import math

        u_lon = unit.get('longitude', 0)
        u_lat = unit.get('latitude', 0)

        e_lon = enemy.get('longitude', 0)
        e_lat = enemy.get('latitude', 0)
        e_alt = enemy.get('altitude', 3000)
        e_heading = math.degrees(enemy.get('heading', 0)) % 360

        # 计算当前姿态角（判断是否已经在尾追位置）
        bearing_enemy_to_me = YxGeoUtils.calculate_bearing(e_lon, e_lat, u_lon, u_lat)
        aspect_angle = abs(bearing_enemy_to_me - e_heading)
        if aspect_angle > 180:
            aspect_angle = 360 - aspect_angle

        # 如果已经在尾追位置（姿态角 > 120°），直接飞向最优距离点
        if aspect_angle > 120:
            # 已在尾追位置，飞向最优攻击距离
            direction = YxGeoUtils.calculate_bearing(u_lon, u_lat, e_lon, e_lat)
            dist_km = optimal_dist / 1000
            lon_off, lat_off = YxGeoUtils.km_to_lon_lat(e_lat, dist_km, (direction + 180) % 360)
            return (e_lat + lat_off, e_lon + lon_off, e_alt)

        # 计算敌机后方方向
        tail_direction = (e_heading + 180) % 360

        # 计算滞后点（敌机后方10km）
        lag_dist_km = self.LAG_PURSUIT_DISTANCE_KM
        lon_off, lat_off = YxGeoUtils.km_to_lon_lat(e_lat, lag_dist_km, tail_direction)

        lag_lon = e_lon + lon_off
        lag_lat = e_lat + lat_off

        return (lag_lat, lag_lon, e_alt)

    def _has_ammo(self, unit) -> bool:
        """检查单位是否有弹药"""
        for weapon in unit.get('weapons', []):
            if weapon.get('quantity', 0) > 0:
                return True
        return False

    # 有人机安全攻击参数
    MANNED_SAFE_ATTACK_DISTANCE = 40000  # 40km内检查威胁
    MANNED_MAX_ARMED_ENEMIES = 1         # 最多允许1架有弹药敌机

    def _should_manned_attack(self, agent, unit) -> bool:
        """
        判断有人机是否应该攻击

        条件：40km内最多只有1架有弹药的敌机
        这确保有人机入场后不会被多架敌机同时威胁

        Args:
            agent: 智能体实例
            unit: 有人机单位

        Returns:
            bool: True表示可以攻击，False表示应该保持后方
        """
        if unit.get('type') != '有人机':
            return True  # 无人机不受此限制

        u_lon = unit.get('longitude', 0)
        u_lat = unit.get('latitude', 0)

        enemies_with_ammo = 0
        for enemy in agent.enemy_units:
            e_lon = enemy.get('longitude', 0)
            e_lat = enemy.get('latitude', 0)
            if not e_lon or not e_lat:
                continue

            dist = YxGeoUtils.haversine_distance(u_lon, u_lat, e_lon, e_lat)
            if dist < self.MANNED_SAFE_ATTACK_DISTANCE:
                # 检查敌机是否有弹药
                if self._enemy_has_ammo(enemy):
                    enemies_with_ammo += 1

        # 有人机只在最多1架有弹药敌机威胁时才攻击
        return enemies_with_ammo <= self.MANNED_MAX_ARMED_ENEMIES

    def _enemy_has_ammo(self, enemy) -> bool:
        """
        检查敌机是否有弹药

        注意：敌机信息可能不完整，需要通过多种方式判断
        """
        # 方式1：直接检查weapons字段（如果有的话）
        weapons = enemy.get('weapons', [])
        if weapons:
            for weapon in weapons:
                if weapon.get('quantity', 0) > 0:
                    return True
            return False

        # 方式2：检查is_fired_num（已发射导弹数）
        # 假设每架飞机初始有4枚导弹（官方参数）
        fired_num = enemy.get('is_fired_num', 0)
        enemy_type = enemy.get('platform_entity_type', '无人机')

        if enemy_type == '有人机':
            max_ammo = 4  # 有人机4枚导弹
        else:
            max_ammo = 2  # 无人机2枚导弹

        remaining = max_ammo - fired_num
        return remaining > 0

    def _calculate_attack_angle_difference(self, unit1, unit2, enemy) -> float:
        """
        计算两架飞机相对目标的攻击角度差

        返回两架飞机从目标视角看的方位角差异（0-180度）
        角度差越大，目标越难同时规避两个方向的攻击
        """
        import math

        # 获取坐标
        u1_lon = unit1.get('longitude', unit1.get('X', 0))
        u1_lat = unit1.get('latitude', unit1.get('Y', 0))
        u2_lon = unit2.get('longitude', unit2.get('X', 0))
        u2_lat = unit2.get('latitude', unit2.get('Y', 0))
        e_lon = enemy.get('longitude', enemy.get('X', 0))
        e_lat = enemy.get('latitude', enemy.get('Y', 0))

        # 计算从目标到两架飞机的方位角
        angle1 = math.atan2(u1_lon - e_lon, u1_lat - e_lat) * 180 / math.pi
        angle2 = math.atan2(u2_lon - e_lon, u2_lat - e_lat) * 180 / math.pi

        # 计算角度差（0-180度）
        diff = abs(angle1 - angle2)
        if diff > 180:
            diff = 360 - diff

        return diff

    def _record_launch_data(self, agent, unit, enemy, cand, pending_key):
        """记录导弹发射数据用于分析"""
        if not self.COLLECT_KILL_DATA:
            return

        u_lon = unit.get('longitude', 0)
        u_lat = unit.get('latitude', 0)
        e_lon = enemy.get('longitude', 0)
        e_lat = enemy.get('latitude', 0)

        target_id = enemy.get('target_id', enemy.get('id', id(enemy)))
        target_name = enemy.get('target_name', enemy.get('name', ''))

        aspect = SmartFireControl.calculate_aspect_angle(
            u_lon, u_lat, unit.get('heading', 0),
            e_lon, e_lat, enemy.get('heading', 0)
        )
        closure = SmartFireControl.calculate_closure_rate(
            u_lon, u_lat, unit.get('speed', 300), unit.get('heading', 0),
            e_lon, e_lat, enemy.get('speed', 300), enemy.get('heading', 0)
        )
        is_manned = unit.get('type') == '有人机'
        nez = SmartFireControl.calculate_nez(is_manned, aspect)

        # 估算导弹飞行距离（基于官方参数：导弹速度1200m/s）
        missile_speed = 1200  # 官方导弹速度，有人机和无人机相同

        agent.missile_launch_data[pending_key] = {
            'launch_frame': agent.frame_count,
            'shooter_name': unit['name'],
            'shooter_type': unit.get('type', '无人机'),
            'target_id': target_id,
            'target_name': target_name,
            'target_type': enemy.get('platform_entity_type', '无人机'),
            'distance': cand['dist'],
            'aspect_angle': aspect,
            'closure_rate': closure,
            'pk': cand['pk'],
            'nez': nez,
            'in_nez': cand['dist'] <= nez,
            'shooter_speed': unit.get('speed', 300),
            'target_speed': enemy.get('speed', 300),
            'shooter_alt': unit.get('altitude', 3000),
            'target_alt': enemy.get('altitude', 3000),
            'missile_speed': missile_speed,
            'altitude_diff': abs(unit.get('altitude', 3000) - enemy.get('altitude', 3000)),
            'combined_speed': closure + missile_speed,
            'coordinated_fire': self.REQUIRE_COORDINATED_FIRE,  # 标记是否为协同开火
        }
