import math
import random, config
from typing import List, Dict, Tuple

from env.agent.agent_base import AutoAgentBase
from utilities.yxScriptTreeFunc import YxScriptTreeFunc as decCmd


class XXXXAutoAgent(AutoAgentBase):
    def __init__(self, side, name):
        super().__init__(side, name)
        # TODO: 代码实现
        # 初始化

    def update_decision(self, new_observation: Dict):
        """
        获取态势，更新己方有人机、无人机决策逻辑
        new_observation: 新获取的我方视角态势数据
        """
        # TODO: 代码实现
        actions = []
        # 态势分析 ...
        # 决策指令生成
        # actions.append(decCmd.xxx) ...
        # 示例：actions.append(decCmd.fire_track("红无人机1", "蓝无人机1"))
        # 示例：actions.append(decCmd.go_to_speed("红有人机1", 400))
        return actions
