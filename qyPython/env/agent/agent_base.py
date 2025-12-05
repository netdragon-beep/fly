from abc import ABC, abstractmethod
from typing import List
from utilities.yxHttp import YxHttpRequest as yxHttp
from utilities.yxRegistry import YxRegistry


class AgentBase(ABC):
    def __init__(self, side, name):
        self.side = side
        self.name = name
        self.observation = None
        self.last_observation = None

        # 作战区域参数（长方形区域，使用经纬度表示）
        self.battlefield = yxHttp.get_task_battlefield()
        # 友军态势
        self.friendly_units = []

    @abstractmethod
    def update_decision(self, new_observation: dict) -> List[dict]:
        """
        :param new_observation: 观察到的态势信息
        :return: 决策列表，每个决策是一个字典，包含动作类型和参数
        """
        pass

    def get_cmds(self, new_observation) -> List[dict]:
        self.init_units(new_observation)
        actions = self.update_decision(new_observation)
        #移除己方执行指定任务的无人机及敌方无人机
        actions = [action for action in actions if
                   action['args'][0] in self.friendly_units]
        return actions

    def init_units(self, observation: dict):
        units = observation.get('platform_list', [])
        for unit in units:
            if unit['side'] == self.side and unit['name'] not in self.friendly_units:
                self.friendly_units.append(unit['name'])


class AutoAgentBase(AgentBase):
    """
    机器自主智能体的基类
    """
    def __init__(self, side, name):
        super().__init__(side, name)

    @abstractmethod
    def update_decision(self, new_observation):
        pass

class TeamingAgentBase(AgentBase):
    """
    人机协同智能体的基类
    """
    def __init__(self, side, name):
        super().__init__(side, name)
        self.controlled_platform = []

    @abstractmethod
    def update_decision(self, new_observation):
        pass

    def get_cmds(self, new_observation) -> List[dict]:
        actions = super().get_cmds(new_observation)
        actions = [action for action in actions if
                   action['args'][0] not in task_platform_list]
        #添加指定执行任务生成的指令
        if len(task_platform_list) > 0:
            task_actions = _do_task(new_observation)
            if len(task_actions):
                actions.append(task_actions)
        return actions

    def init_units(self, observation: dict):
        super().init_units(observation)
        if len(self.controlled_platform) > 0:
            return
        units = observation.get('platform_list', [])
        for unit in units:
            if unit['side'] == self.side and unit['type'] == '有人机':
                self.controlled_platform.append(unit['name'])

@YxRegistry.register("cancelTask", -1)
def release_platform(target, raw_observation, web_call=False):
    global task_platform_list
    if target in task_platform_list:
        task_platform_list.remove(target)
        task_dict.pop(target)

def _add_task(targets, func):
    global task_dict
    for target in targets:
        task_dict[target] = func

def _base_function(targets, func):
    global task_platform_list
    for target in targets:
        if target not in task_platform_list:
            task_platform_list.append(target)
        _add_task(targets, func)

def teaming_task(func):
    def wrapper(*args):
        _base_function(args[0], func)
    return wrapper

def _do_task(raw_observation) -> List[dict]:
    actions = []
    func_targets = {}
    try:
        for target, func in task_dict.items():
            if func in func_targets:
                func_targets[func].append(target)
            else:
                func_targets[func] = [target]
        for func, targets in func_targets.items():
            t_actions = func(targets, raw_observation)
            if len(t_actions) > 0:
                for action in t_actions:
                    actions.append(action)
    except Exception as e:
        print(f"do_task error: {e}")
    return actions

task_platform_list = []
task_dict = {}