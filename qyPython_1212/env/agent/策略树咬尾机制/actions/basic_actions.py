"""
基础动作节点

包含：
- ActionResetFrame: 每帧重置
- ConditionCheckInitialDeployment: 检查初始部署
- ActionExecuteDeployment: 执行开局部署
- ActionMannedFollowConstraint: 有人机领航跟随约束（强制）
"""

import math
from ..bt_framework import Action, Condition, NodeStatus
from utilities.yxScriptTreeFunc import YxScriptTreeFunc as decCmd
from utilities.yxGeoUtils import YxGeoUtils


class ActionResetFrame(Action):
    """每一帧开始前的清理工作"""
    def tick(self, agent) -> str:
        agent.current_actions = []        # 清空指令列表
        agent.commanded_units = set()     # 清空已被占用的单位集合
        agent.missile_threats = []        # 清空威胁缓存
        return NodeStatus.SUCCESS


class ActionMannedFollowConstraint(Action):
    """
    有人机领航跟随约束（Leader-Follower Algorithm）

    这是一个强制约束节点，放在行为树最前面执行。
    核心思想：有人机必须始终保持在无人机后方，这是硬性约束。

    算法原理（基于领航-跟随编队控制）：
    1. 无人机群作为"领航者"（Leader）
    2. 有人机作为"跟随者"（Follower）
    3. 跟随者必须保持与领航者的相对距离 d >= D_min
    4. 如果距离不足，跟随者必须减速或后撤

    约束检查（每帧执行）：
    - 计算有人机与无人机阵线的相对位置
    - 如果有人机超前或距离不足 → 强制盘旋等待
    - 只有距离足够时，才允许其他节点控制有人机

    参考文献：
    - Leader-Follower Formation Control (Nature Scientific Reports, 2024)
    - 基于分布式模型预测控制的无人机编队控制
    """

    # 领航跟随参数
    MIN_FOLLOW_DISTANCE_KM = 25     # 最小跟随距离 25km
    SAFE_FOLLOW_DISTANCE_KM = 35    # 安全跟随距离 35km
    LOITER_RADIUS_KM = 5            # 盘旋半径 5km
    LOITER_SPEED = 220              # 盘旋速度 m/s（低速等待）
    LOITER_ALTITUDE = 4000          # 盘旋高度 m

    # 残局判定（残局时解除约束）
    ENDGAME_UNARMED_RATIO = 0.5     # 敌方超过50%无弹药

    def tick(self, agent) -> str:
        """
        每帧检查有人机是否满足跟随约束

        返回 SUCCESS 继续执行后续节点
        但如果有人机违反约束，会先给有人机发盘旋指令
        """
        manned_units = [u for u in agent.own_units if u.get('type') == '有人机']
        uavs = [u for u in agent.own_units if u.get('type') == '无人机']

        if not manned_units:
            return NodeStatus.SUCCESS

        # 检查是否进入残局（残局时解除约束，允许有人机追击）
        if self._is_endgame(agent):
            return NodeStatus.SUCCESS

        # 如果没有无人机，有人机需要自己判断
        if not uavs:
            return NodeStatus.SUCCESS

        # 计算无人机阵线位置（领航者位置）
        uav_avg_lon = sum(u.get('longitude', 0) for u in uavs) / len(uavs)
        uav_avg_lat = sum(u.get('latitude', 0) for u in uavs) / len(uavs)

        # 判断敌方方向
        enemy_dir = 1 if agent.side == 'red' else -1

        for unit in manned_units:
            manned_lon = unit.get('longitude', 0)
            manned_lat = unit.get('latitude', 0)

            # 计算有人机相对于无人机阵线的位置
            # 正数 = 在无人机前方（危险！）
            # 负数 = 在无人机后方（安全）
            relative_pos_km = (manned_lon - uav_avg_lon) * enemy_dir * 111.0

            # 检查是否违反跟随约束
            if relative_pos_km > -self.MIN_FOLLOW_DISTANCE_KM:
                # 违反约束！有人机太靠前了
                # 强制盘旋等待，并标记为已占用
                self._force_loiter(agent, unit, manned_lat, manned_lon)

        return NodeStatus.SUCCESS

    def _is_endgame(self, agent) -> bool:
        """检查是否进入残局（敌方多数无弹药）"""
        if not agent.enemy_units:
            return True  # 无敌机，算残局

        total = len(agent.enemy_units)
        unarmed = 0

        for enemy in agent.enemy_units:
            if not self._enemy_has_ammo(enemy):
                unarmed += 1

        return (unarmed / total) >= self.ENDGAME_UNARMED_RATIO

    def _enemy_has_ammo(self, enemy) -> bool:
        """检查敌机是否有弹药"""
        weapons = enemy.get('weapons', [])
        if weapons:
            for w in weapons:
                if w.get('quantity', 0) > 0:
                    return True
            return False

        # 没有武器信息，根据已发射数量估算
        fired = enemy.get('is_fired_num', 0)
        enemy_type = enemy.get('platform_entity_type', '无人机')
        max_ammo = 4 if enemy_type == '有人机' else 2
        return (max_ammo - fired) > 0

    def _force_loiter(self, agent, unit, lat, lon):
        """强制有人机原地盘旋等待"""
        unit_name = unit.get('name', '')

        # 盘旋角度（每帧递增）
        if not hasattr(agent, '_follow_loiter_angle'):
            agent._follow_loiter_angle = 0
        agent._follow_loiter_angle = (agent._follow_loiter_angle + 5) % 360

        # 计算盘旋目标点
        radius_deg = self.LOITER_RADIUS_KM / 111.0
        angle_rad = math.radians(agent._follow_loiter_angle)

        target_lat = lat + radius_deg * math.sin(angle_rad)
        target_lon = lon + radius_deg * math.cos(angle_rad)

        # 发送盘旋指令并标记为已占用
        agent.add_action(
            decCmd.fly_to_point(
                unit_name,
                (target_lat, target_lon, self.LOITER_ALTITUDE),
                self.LOITER_SPEED
            ),
            unit_name  # 标记为已占用，后续节点不会再控制
        )


class ConditionCheckInitialDeployment(Condition):
    """检查是否完成了初始部署"""
    def tick(self, agent) -> str:
        if not agent.initial_deployment_complete:
            return NodeStatus.SUCCESS  # 需要部署
        return NodeStatus.FAILURE      # 不需要部署


class ActionExecuteDeployment(Action):
    """
    执行开局部署

    策略：
    - 无人机立即前推，速度拉满360m/s，建立视野
    - 有人机原地盘旋等待，等无人机前推后再跟进
    - 盘旋半径约5km，确保有人机不会冲到无人机前面
    """

    # 官方参数
    MANNED_MAX_SPEED = 500   # 有人机最大速度 m/s
    UAV_MAX_SPEED = 360      # 无人机最大速度 m/s

    # 有人机盘旋参数
    MANNED_LOITER_RADIUS_KM = 5    # 盘旋半径 km
    MANNED_LOITER_SPEED = 250      # 盘旋速度 m/s（低速盘旋）
    MANNED_LOITER_ALT = 4000       # 盘旋高度 m

    def tick(self, agent) -> str:
        # 分离有人机和无人机
        manned = [u for u in agent.own_units if u.get('type') == '有人机']
        uavs = [u for u in agent.own_units if u.get('type') == '无人机']

        # 无人机立即前推，速度拉满360m/s
        for unit in uavs:
            offset_lon = 0.45 if agent.side == 'red' else -0.45
            target = (unit['latitude'], unit['longitude'] + offset_lon, 4000)
            agent.add_action(decCmd.fly_to_point(unit['name'], target, self.UAV_MAX_SPEED), unit['name'])

        # 有人机原地盘旋等待（不前推！）
        # 盘旋点设在当前位置，让无人机先建立视野
        for unit in manned:
            # 计算盘旋起始点（当前位置稍微偏移形成圆形轨迹）
            unit_lat = unit['latitude']
            unit_lon = unit['longitude']

            # 盘旋半径转经纬度
            radius_deg = self.MANNED_LOITER_RADIUS_KM / 111.0

            # 初始盘旋角度（随机起点避免多架有人机重叠）
            if not hasattr(agent, 'manned_loiter_start_angle'):
                agent.manned_loiter_start_angle = 0

            angle_rad = math.radians(agent.manned_loiter_start_angle)
            loiter_lat = unit_lat + radius_deg * math.sin(angle_rad)
            loiter_lon = unit_lon + radius_deg * math.cos(angle_rad)

            target = (loiter_lat, loiter_lon, self.MANNED_LOITER_ALT)
            agent.add_action(decCmd.fly_to_point(unit['name'], target, self.MANNED_LOITER_SPEED), unit['name'])

        agent.initial_deployment_complete = True
        return NodeStatus.SUCCESS
