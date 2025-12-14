import config, random, math
from typing import List, Dict, Tuple
from env.agent.agent_base import TeamingAgentBase, teaming_task
from utilities.yxRegistry import YxRegistry
from utilities.yxScriptTreeFunc import YxScriptTreeFunc as callFunc


class XXXXTeamingAgent(TeamingAgentBase):
    def __init__(self, side, name):
        super().__init__(side, name)
        # TODO: 代码实现
        # 初始化


    def update_decision(self, new_observation: dict):
        """
        获取态势，更新己方无人机决策逻辑
        new_observation: 新获取的我方视角态势数据
        """
        # TODO: 代码实现
        actions = []
        # 态势分析 ...
        # 无人机决策指令生成
        # actions.append(decCmd.xxx) ...
        # 示例：actions.append(decCmd.fire_track("红无人机1", "蓝无人机1"))
        # 示例：actions.append(decCmd.go_to_speed("红无人机1", 500))
        return actions


@YxRegistry.register(name="单平台任务示例", func_type=0)
@teaming_task
def task1(selected_forces: List[str], raw_observation: Dict):
    print(f"{selected_forces}执行单平台任务")
    actions = []
    # TODO: 单平台任务决策逻辑
    return actions

@YxRegistry.register(name="编组任务示例", func_type=1)
@teaming_task
def task2(selected_forces: List[str], raw_observation: Dict):
    print(f"{selected_forces}执行编组任务")
    actions = []
    # TODO: 编组任务决策逻辑
    return actions

@YxRegistry.register(name="通用任务示例", func_type=2)
@teaming_task
def task3(selected_forces: List[str], raw_observation: Dict):
    print(f"{selected_forces}执行通用任务")
    actions = []
    # TODO: 通用任务决策逻辑
    return actions