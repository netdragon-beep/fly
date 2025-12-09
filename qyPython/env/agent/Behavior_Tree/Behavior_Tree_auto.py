import math
import random
import config
from typing import List, Dict, Tuple, Set
from env.agent.agent_base import AutoAgentBase
from utilities.yxScriptTreeFunc import YxScriptTreeFunc as decCmd
from utilities.yxGeoUtils import YxGeoUtils

# ==========================================
# Part 1: 轻量级行为树框架 (Mini BT Engine)
# ==========================================

class BTNode:
    """行为树节点基类"""
    def tick(self, agent) -> str:
        raise NotImplementedError

class NodeStatus:
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"

class Sequence(BTNode):
    """序列节点 (AND)：所有子节点成功才算成功，按顺序执行"""
    def __init__(self, children: List[BTNode]):
        self.children = children

    def tick(self, agent) -> str:
        for child in self.children:
            status = child.tick(agent)
            if status != NodeStatus.SUCCESS:
                return status
        return NodeStatus.SUCCESS

class Selector(BTNode):
    """选择节点 (OR)：只要有一个子节点成功就算成功"""
    def __init__(self, children: List[BTNode]):
        self.children = children

    def tick(self, agent) -> str:
        for child in self.children:
            status = child.tick(agent)
            if status == NodeStatus.SUCCESS:
                return NodeStatus.SUCCESS
            if status == NodeStatus.RUNNING:
                return NodeStatus.RUNNING
        return NodeStatus.FAILURE

class Action(BTNode):
    """动作节点基类"""
    pass

class Condition(BTNode):
    """条件节点基类"""
    pass

# ==========================================
# Part 2: 具体的业务逻辑节点 (Leaf Nodes)
# ==========================================

class ActionResetFrame(Action):
    """每一帧开始前的清理工作"""
    def tick(self, agent) -> str:
        agent.current_actions = []        # 清空指令列表
        agent.commanded_units = set()     # 清空已被占用的单位集合
        agent.missile_threats = []        # 清空威胁缓存
        return NodeStatus.SUCCESS

class ConditionCheckInitialDeployment(Condition):
    """检查是否完成了初始部署"""
    def tick(self, agent) -> str:
        if not agent.initial_deployment_complete:
            return NodeStatus.SUCCESS # 需要部署
        return NodeStatus.FAILURE     # 不需要部署

class ActionExecuteDeployment(Action):
    """执行开局部署"""
    def tick(self, agent) -> str:
        # 分离有人机和无人机
        manned = [u for u in agent.own_units if u.get('type') == '有人机']
        uavs = [u for u in agent.own_units if u.get('type') == '无人机']
        
        # 无人机靠前
        for unit in uavs:
            offset_lon = 0.45 if agent.side == 'red' else -0.45
            target = (unit['latitude'], unit['longitude'] + offset_lon, 3500)
            agent.add_action(decCmd.fly_to_point(unit['name'], target, 400), unit['name'])
            
        # 有人机靠后
        for unit in manned:
            offset_lon = 0.27 if agent.side == 'red' else -0.27
            target = (unit['latitude'], unit['longitude'] + offset_lon, 4000)
            agent.add_action(decCmd.fly_to_point(unit['name'], target, 300), unit['name'])

        agent.initial_deployment_complete = True
        return NodeStatus.SUCCESS

class ActionEvadeMissiles(Action):
    """高优先级：规避导弹"""
    def tick(self, agent) -> str:
        missile_avoided_this_frame = set()
        
        for missile in agent.enemy_missiles:
            missile_id = missile.get('target_id') or id(missile)
            
            # 简单的空间关系判断威胁
            threatened_units = agent.predict_threatened_units(missile)
            
            for unit in threatened_units:
                # 如果该单位已经被指令规避了（可能是另一枚导弹），跳过
                if unit['name'] in agent.commanded_units:
                    continue
                    
                # 计算规避点
                evade_dir = agent.calculate_evade_direction(unit, missile)
                lon_off, lat_off = YxGeoUtils.km_to_lon_lat(agent.center_lat, 5, evade_dir)
                
                evade_point = (
                    unit.get('latitude') + lat_off,
                    unit.get('longitude') + lon_off,
                    unit.get('altitude', 3000) + 500
                )
                
                # 添加指令并标记单位被占用
                agent.add_action(decCmd.fly_to_point(unit['name'], evade_point, 700), unit['name'])
                missile_avoided_this_frame.add(missile_id)
                
        return NodeStatus.SUCCESS

class ActionAttackLogic(Action):
    """攻击逻辑：包含开火和机动"""
    def tick(self, agent) -> str:
        if not agent.enemy_units:
            return NodeStatus.SUCCESS

        # Sub-step 1: Fire Control (所有单位都可以开火，哪怕正在规避)
        # 计算距离矩阵
        dist_pairs = []
        for unit in agent.own_units:
            for enemy in agent.enemy_units:
                d = YxGeoUtils.haversine_distance(
                    unit['longitude'], unit['latitude'],
                    enemy['longitude'], enemy['latitude']
                )
                dist_pairs.append({'unit': unit, 'enemy': enemy, 'dist': d})
        
        dist_pairs.sort(key=lambda x: x['dist']) # 距离排序
        
        cur_round_shots = {} # 本轮发射记录
        
        for pair in dist_pairs:
            u, e, d = pair['unit'], pair['enemy'], pair['dist']
            
            # 攻击条件检查
            # 1. 敌机未被过度攻击
            already_fired = e.get('is_fired_num', 0) + cur_round_shots.get(e['target_id'], 0)
            if already_fired >= 4: continue
            
            # 2. 射程检查 (90%射程)
            max_range = 20000 if u['type'] == '有人机' else 15000
            if d > max_range * 0.9: continue
            
            # 3. 弹药检查与发射
            # 注意：fire_track 不会占用单位的运动控制权，所以不需要 check commanded_units
            if agent.try_fire_weapon(u):
                agent.add_action(decCmd.fire_track(u['name'], e['target_name']), None) # None表示不占用移动权
                cur_round_shots[e['target_id']] = cur_round_shots.get(e['target_id'], 0) + 1

        # Sub-step 2: Maneuver to Attack (仅对未被占用的单位有效)
        for unit in agent.own_units:
            if unit['name'] in agent.commanded_units:
                continue # 正在规避的单位不执行进攻机动
            
            # 找最近敌机
            closest = None
            min_d = float('inf')
            for enemy in agent.enemy_units:
                d = YxGeoUtils.haversine_distance(
                    unit['longitude'], unit['latitude'],
                    enemy['longitude'], enemy['latitude']
                )
                if d < min_d:
                    min_d = d
                    closest = enemy
            
            if closest:
                # 飞向攻击占位点
                optimal_dist = 18000 if unit['type'] == '有人机' else 13500
                direction = YxGeoUtils.calculate_direction_to(
                    closest['longitude'], closest['latitude'],
                    unit['longitude'], unit['latitude']
                )
                lon_off, lat_off = YxGeoUtils.km_to_lon_lat(agent.center_lat, optimal_dist/1000, direction)
                
                target_pt = (
                    closest['latitude'] + lat_off,
                    closest['longitude'] + lon_off,
                    closest.get('altitude', 3000)
                )
                agent.add_action(decCmd.fly_to_point(unit['name'], target_pt, 500), unit['name'])

        return NodeStatus.SUCCESS

class ActionPatrolFormation(Action):
    """防御/巡逻阵型"""
    def tick(self, agent) -> str:
        # 仅控制剩下的单位
        available_units = [u for u in agent.own_units if u['name'] not in agent.commanded_units]
        if not available_units:
            return NodeStatus.SUCCESS
            
        manned = [u for u in available_units if u.get('type') == '有人机']
        uavs = [u for u in available_units if u.get('type') == '无人机']
        
        # 内环有人机
        for i, unit in enumerate(manned):
            angle = (i / len(manned) * 360 + agent.defense_angle_offset) % 360
            lon_off, lat_off = YxGeoUtils.km_to_lon_lat(agent.center_lat, 15, angle)
            pt = (agent.center_lat + lat_off, agent.center_lon + lon_off, 4000)
            agent.add_action(decCmd.fly_to_point(unit['name'], pt, 300), unit['name'])
            
        # 外环无人机
        for i, unit in enumerate(uavs):
            angle = (i / len(uavs) * 360 + agent.defense_angle_offset) % 360
            lon_off, lat_off = YxGeoUtils.km_to_lon_lat(agent.center_lat, 30, angle)
            pt = (agent.center_lat + lat_off, agent.center_lon + lon_off, 3500)
            agent.add_action(decCmd.fly_to_point(unit['name'], pt, 300), unit['name'])
            
        return NodeStatus.SUCCESS

# ==========================================
# Part 3: 主智能体类 (Agent)
# ==========================================

class BTDemoAgent(AutoAgentBase):
    def __init__(self, side, name):
        super().__init__(side, name)
        
        # --- 状态变量 ---
        self.initial_deployment_complete = False
        self.frame_count = 0
        self.own_units = []
        self.enemy_units = []
        self.enemy_missiles = []
        
        # --- 行为树上下文 (Blackboard) ---
        self.current_actions = []      # 本帧生成的指令列表
        self.commanded_units = set()   # 本帧已分配移动任务的单位名称
        
        # --- 战场辅助信息 ---
        self.center_lat = (self.battlefield['min_lat'] + self.battlefield['max_lat']) / 2
        self.center_lon = (self.battlefield['min_lon'] + self.battlefield['max_lon']) / 2
        self.defense_angle_offset = random.randint(0, 360)

        # --- 构建行为树 ---
        # 逻辑：
        # 1. 优先检查是否需要开局部署。
        # 2. 如果不需要部署，进入战斗循环 (Sequence)：
        #    a. 重置帧数据
        #    b. 处理导弹规避 (占用受威胁单位)
        #    c. 处理攻击 (占用剩余单位，但全员可开火)
        #    d. 处理剩余闲置单位 (巡逻)
        
        self.bt_root = Selector([
            # 分支 1: 开局部署
            Sequence([
                ConditionCheckInitialDeployment(),
                ActionExecuteDeployment()
            ]),
            
            # 分支 2: 常规战斗循环 (Main Loop)
            Sequence([
                ActionResetFrame(),      # 步骤1: 清理
                ActionEvadeMissiles(),   # 步骤2: 生存 (最高级)
                ActionAttackLogic(),     # 步骤3: 战斗
                ActionPatrolFormation()  # 步骤4: 兜底
            ])
        ])

    def update_decision(self, new_observation: Dict):
        """主入口函数"""
        self.observation = new_observation
        self.frame_count += 1
        
        # 1. 解析数据
        self._parse_observation(new_observation)
        
        if not self.own_units:
            return []

        # 2. 运行行为树
        self.bt_root.tick(self)
        
        # 3. 返回行为树生成的指令列表
        return self.current_actions

    # --- 辅助方法 (给节点调用) ---

    def add_action(self, cmd_dict, unit_name_occupy=None):
        """添加指令到列表，并可选地标记单位为'已占用'"""
        self.current_actions.append(cmd_dict)
        if unit_name_occupy:
            self.commanded_units.add(unit_name_occupy)

    def _parse_observation(self, obs):
        self.own_units = obs.get('platform_list', [])
        self.enemy_units = []
        self.enemy_missiles = []
        for track in obs.get('track_list', []):
            if track.get('platform_entity_side') != self.side:
                if track.get('platform_entity_type') == '导弹':
                    self.enemy_missiles.append(track)
                else:
                    self.enemy_units.append(track)

    def try_fire_weapon(self, unit):
        """尝试扣除武器库存，成功返回True"""
        for weapon in unit.get('weapons', []):
            if weapon.get('quantity', 0) > 0:
                weapon['quantity'] -= 1
                return True
        return False

    def predict_threatened_units(self, missile):
        """预测威胁 (移植自原代码)"""
        threatened = []
        m_lon, m_lat = missile['longitude'], missile['latitude']
        m_heading = (math.degrees(missile.get('heading', 0)) + 360) % 360
        
        for unit in self.own_units:
            dist = YxGeoUtils.haversine_distance(unit['longitude'], unit['latitude'], m_lon, m_lat)
            bearing = YxGeoUtils.calculate_bearing(m_lon, m_lat, unit['longitude'], unit['latitude'])
            angle_diff = abs((bearing - m_heading + 180) % 360 - 180) # 归一化角度差
            
            if dist < 10000 and angle_diff < 15:
                threatened.append(unit)
        return threatened

    def calculate_evade_direction(self, unit, missile):
        """计算规避方向 (移植自原代码)"""
        incoming_bearing = YxGeoUtils.calculate_bearing(
            missile['longitude'], missile['latitude'],
            unit['longitude'], unit['latitude']
        )
        # 简单判据：在战场上半区向下转，下半区向上转
        if unit['latitude'] < self.center_lat:
            return (incoming_bearing + 90) % 360
        else:
            return (incoming_bearing - 90) % 360