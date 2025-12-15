"""
行为树框架 (Mini BT Engine)

核心类:
- BTNode: 行为树节点基类
- NodeStatus: 节点状态枚举
- Sequence: 序列节点 (AND)
- Selector: 选择节点 (OR)
- Action: 动作节点基类
- Condition: 条件节点基类
"""

from typing import List


class NodeStatus:
    """节点状态枚举"""
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"


class BTNode:
    """行为树节点基类"""
    def tick(self, agent) -> str:
        raise NotImplementedError


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
