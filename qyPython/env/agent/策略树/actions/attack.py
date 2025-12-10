"""
攻击逻辑动作节点

包含智能火控和Shoot-Look-Shoot策略
支持规则(v1)和强化学习(SAC)两种火控模式
"""

from ..bt_framework import Action, NodeStatus
from ..fire_control import SmartFireControl
from utilities.yxScriptTreeFunc import YxScriptTreeFunc as decCmd
from utilities.yxGeoUtils import YxGeoUtils

# ==================== 火控模式配置 ====================
# 可选: 'rule' (v1规则), 'rl' (SAC强化学习), 'hybrid' (混合)
FIRE_CONTROL_MODE = 'rl'  # 切换到RL模式测试

# RL模型路径 (当使用rl或hybrid模式时需要)
import os
_current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RL_MODEL_PATH = os.path.join(_current_dir, 'checkpoints', 'sac_fire_control', 'sac_fire_control_pretrained.pt')

# 全局火控系统实例 (延迟初始化)
_hybrid_fire_control = None
_rl_import_failed = False  # 标记是否已经尝试导入失败

def get_fire_control():
    """获取火控系统实例"""
    global _hybrid_fire_control, _rl_import_failed

    # 如果已经导入失败过，不再重复尝试
    if _rl_import_failed:
        return None

    if _hybrid_fire_control is None:
        try:
            # 添加父目录到路径
            import sys
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)

            # 直接导入
            from QC1.fly.qyPython.env.agent.策略树.fire_control_rl暂时不打算使用 import HybridFireControl
            _hybrid_fire_control = HybridFireControl(RL_MODEL_PATH)
            _hybrid_fire_control.set_mode(FIRE_CONTROL_MODE)
            print(f"[火控] RL模块加载成功，模式: {FIRE_CONTROL_MODE}")
        except Exception as e:
            print(f"[火控] RL模块导入失败: {e}, 使用规则模式")
            _rl_import_failed = True
            _hybrid_fire_control = None
    return _hybrid_fire_control


def set_fire_control_mode(mode: str, model_path: str = None):
    """
    设置火控模式

    Args:
        mode: 'rule' (v1规则), 'rl' (SAC强化学习), 'hybrid' (混合)
        model_path: RL模型路径 (当mode='rl'或'hybrid'时需要)

    使用示例:
        from env.agent.策略树.actions.attack import set_fire_control_mode

        # 使用规则模式 (默认)
        set_fire_control_mode('rule')

        # 使用RL模式
        set_fire_control_mode('rl', 'checkpoints/sac_fire_control/model.pt')

        # 使用混合模式 (规则+RL都同意才开火)
        set_fire_control_mode('hybrid', 'checkpoints/sac_fire_control/model.pt')
    """
    global FIRE_CONTROL_MODE, RL_MODEL_PATH, _hybrid_fire_control

    FIRE_CONTROL_MODE = mode
    if model_path:
        RL_MODEL_PATH = model_path

    # 重置火控实例，下次调用时重新初始化
    _hybrid_fire_control = None

    print(f"[火控] 模式切换为: {mode}")
    if model_path:
        print(f"[火控] RL模型: {model_path}")


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
    DEBUG_INTERVAL = 10  # 每N帧输出一次火控信息（更频繁）

    # Shoot-Look-Shoot 参数
    MISSILE_FLIGHT_TIME_ESTIMATE = 80  # 估计导弹飞行时间（帧），约8秒@10fps
    MIN_REFIRE_INTERVAL = 40           # 最小再次发射间隔（帧），约4秒
    MAX_PENDING_MISSILES_PER_TARGET = 1  # 每个目标最多同时有N枚待定导弹（防止多射手浪费）

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
                    print(f"[数据收集] 击杀! dist={record['distance']:.0f}m, "
                          f"aspect={record['aspect_angle']:.1f}°, "
                          f"closure={record['closure_rate']:.1f}m/s, Pk={record['pk']:.2f}, "
                          f"飞行={elapsed}帧, 估算拦截距离={record['est_intercept_distance']:.0f}m")

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

        # 先收集所有有效的 (unit, enemy) 对
        valid_pairs = []
        pair_metadata = []  # 存储额外信息

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

                valid_pairs.append((unit, enemy))
                # 计算距离用于排序
                d = YxGeoUtils.haversine_distance(u_lon, u_lat, e_lon, e_lat)
                # 判断敌机类型：有人机优先级更高
                enemy_type = enemy.get('platform_entity_type', '无人机')
                priority = 0 if enemy_type == '有人机' else 1
                pair_metadata.append({'dist': d, 'priority': priority})

        # 批量评估火控决策
        hybrid_fc = get_fire_control()
        if hybrid_fc and FIRE_CONTROL_MODE != 'rule' and valid_pairs:
            # 使用批量推理 (高效)
            results = hybrid_fc.batch_should_fire(valid_pairs, agent)
            for i, (unit, enemy) in enumerate(valid_pairs):
                should_fire, pk, reason = results[i]
                fire_candidates.append({
                    'unit': unit,
                    'enemy': enemy,
                    'dist': pair_metadata[i]['dist'],
                    'priority': pair_metadata[i]['priority'],
                    'should_fire': should_fire,
                    'pk': pk,
                    'reason': reason
                })
        else:
            # 规则模式，逐个评估
            for i, (unit, enemy) in enumerate(valid_pairs):
                should_fire, pk, reason = SmartFireControl.should_fire(unit, enemy, agent)
                fire_candidates.append({
                    'unit': unit,
                    'enemy': enemy,
                    'dist': pair_metadata[i]['dist'],
                    'priority': pair_metadata[i]['priority'],
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
            print(f"[火控系统] Frame {agent.frame_count}")
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

            # 射击方案评估
            if fire_candidates:
                print(f"[射击方案评估] 共 {len(fire_candidates)} 个")
                for i, cand in enumerate(fire_candidates[:8]):
                    unit_name = cand['unit']['name']
                    enemy_name = cand['enemy'].get('target_name', cand['enemy'].get('name', '?'))
                    status = "✓开火" if cand['should_fire'] else "✗等待"
                    print(f"  {i+1}. {unit_name} -> {enemy_name}: "
                          f"Pk={cand['pk']:.2f}, 距离={cand['dist']/1000:.1f}km, "
                          f"{status}, {cand['reason']}")

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

            # 检查1: 该目标已有多少枚待定导弹（来自所有射手）
            missiles_to_target = sum(1 for key in agent.pending_missiles if key[1] == target_id)
            if missiles_to_target >= self.MAX_PENDING_MISSILES_PER_TARGET:
                if self.DEBUG_ENABLED and should_debug:
                    shooters = [key[0] for key in agent.pending_missiles if key[1] == target_id]
                    print(f"  [全局限制] 目标{target_id}已有{missiles_to_target}枚导弹在途 "
                          f"(射手: {', '.join(shooters)}), {u['name']}跳过")
                continue

            # 检查2: 该射手是否已有导弹正在飞向该目标
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

                # === 收集发射时的详细参数 ===
                u_lon = u.get('longitude', 0)
                u_lat = u.get('latitude', 0)
                e_lon = e.get('longitude', 0)
                e_lat = e.get('latitude', 0)

                aspect = SmartFireControl.calculate_aspect_angle(
                    u_lon, u_lat, u.get('heading', 0),
                    e_lon, e_lat, e.get('heading', 0)
                )
                closure = SmartFireControl.calculate_closure_rate(
                    u_lon, u_lat, u.get('speed', 300), u.get('heading', 0),
                    e_lon, e_lat, e.get('speed', 300), e.get('heading', 0)
                )
                is_manned = u.get('type') == '有人机'
                target_is_manned = e.get('platform_entity_type') == '有人机'
                nez = SmartFireControl.calculate_nez(is_manned, aspect)

                # === 数据收集：记录发射参数 ===
                if self.COLLECT_KILL_DATA:
                    # 估算导弹飞行距离（用于分析伤害衰减）
                    # 无人机导弹速度约800m/s，有人机约900m/s
                    missile_speed = 900 if is_manned else 800

                    agent.missile_launch_data[pending_key] = {
                        'launch_frame': agent.frame_count,
                        'shooter_name': u['name'],
                        'shooter_type': u.get('type', '无人机'),
                        'target_id': target_id,
                        'target_name': target_name,
                        'target_type': e.get('platform_entity_type', '无人机'),
                        'distance': cand['dist'],
                        'aspect_angle': aspect,
                        'closure_rate': closure,
                        'pk': cand['pk'],
                        'nez': nez,
                        'in_nez': cand['dist'] <= nez,
                        'shooter_speed': u.get('speed', 300),
                        'target_speed': e.get('speed', 300),
                        'shooter_alt': u.get('altitude', 3000),
                        'target_alt': e.get('altitude', 3000),
                        # 新增：用于分析伤害衰减
                        'missile_speed': missile_speed,
                        'altitude_diff': abs(u.get('altitude', 3000) - e.get('altitude', 3000)),
                        'combined_speed': closure + missile_speed,  # 导弹相对目标的逼近速度
                    }

                if self.DEBUG_ENABLED:
                    print(f"\n[开火!!!] {u['name']} -> {target_name}")
                    print(f"  距离: {cand['dist']/1000:.2f}km, NEZ: {nez/1000:.2f}km")
                    print(f"  姿态角: {aspect:.1f}° (0=迎头, 180=尾追)")
                    print(f"  接近率: {closure:.1f}m/s (正=接近, 负=远离)")
                    print(f"  Pk: {cand['pk']:.2f}, 原因: {cand['reason']}")

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
        cls.print_kill_data_summary(agent.kill_data_records)

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
