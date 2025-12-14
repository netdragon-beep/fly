"""
基础动作节点

包含：
- ActionResetFrame: 每帧重置
- ConditionCheckInitialDeployment: 检查初始部署
- ActionExecuteDeployment: 执行开局部署
"""

from ..bt_framework import Action, Condition, NodeStatus
from utilities.yxScriptTreeFunc import YxScriptTreeFunc as decCmd


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
            return NodeStatus.SUCCESS  # 需要部署
        return NodeStatus.FAILURE      # 不需要部署


class ActionExecuteDeployment(Action):
    """
    执行开局部署
    """

    # 官方参数
    MANNED_MAX_SPEED = 500   # 有人机最大速度 m/s
    UAV_MAX_SPEED = 360      # 无人机最大速度 m/s

    def tick(self, agent) -> str:
        # 分离有人机和无人机
        manned = [u for u in agent.own_units if u.get('type') == '有人机']
        uavs = [u for u in agent.own_units if u.get('type') == '无人机']

        # 无人机靠前，速度拉满360m/s
        for unit in uavs:
            offset_lon = 0.45 if agent.side == 'red' else -0.45
            target = (unit['latitude'], unit['longitude'] + offset_lon, 4000)
            agent.add_action(decCmd.fly_to_point(unit['name'], target, self.UAV_MAX_SPEED), unit['name'])

        # 有人机靠后，速度拉满500m/s
        for unit in manned:
            offset_lon = 0.27 if agent.side == 'red' else -0.27
            target = (unit['latitude'], unit['longitude'] + offset_lon, 4000)
            agent.add_action(decCmd.fly_to_point(unit['name'], target, self.MANNED_MAX_SPEED), unit['name'])

        agent.initial_deployment_complete = True
        return NodeStatus.SUCCESS
