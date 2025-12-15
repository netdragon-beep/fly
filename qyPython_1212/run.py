import config
from env.env import auto_engage_main, teaming_engage_main
from env.multi_env import auto_engage_main_multi
from env.agent.demo.demo_auto_agent import DemoAutoAgent
from env.agent.demo.demo_teaming_agent import DemoTeamingAgent
from env.agent.策略树.Behavior_Tree_auto import BTDemoAgent  # 行为树智能体（学长的垂直躲避策略）
from env.agent.策略树躲避.Behavior_Tree_auto import BTDemoAgent as BTDemoAgentCopy  # 策略树copy
# from env.agent.test1.test1_auto_agent import FlyTeamAutoAgent
# from env.agent.test1.test1_teaming_agent import FlyTeamTeamingAgent
from utilities.yxHttp import YxHttpRequest as yxHttp

# ========== 红方智能体选择 ==========
# 0: Demo（官方示例）
# 1: 行为树（学长更新的垂直躲避策略）
# 2: 策略树copy（含SSRL规避）
RED_AGENT_MODE = 2
# ==================================

# ========== 蓝方智能体选择 ==========
# 0: Demo（官方示例）
# 1: 策略树copy
# 2: 行为树（学长更新的垂直躲避策略）
BLUE_AGENT_MODE = 2
# ==================================

if __name__ == '__main__':

    yxHttp.clear_room()
    if config.current_config.battle_mode == config.ENGAGE_MODE_AUTO:
        # 机器竞技模式

        # 红方根据配置选择
        if RED_AGENT_MODE == 0:
            red_agent = DemoAutoAgent('red', "red_demo")
            print("[配置] 红方使用: Demo")
        elif RED_AGENT_MODE == 1:
            red_agent = BTDemoAgent('red', "red_bt")
            print("[配置] 红方使用: 行为树（垂直躲避）")
        else:
            red_agent = BTDemoAgentCopy('red', "red_bt_copy")
            print("[配置] 红方使用: 策略树copy（含SSRL规避）")

        # 蓝方根据配置选择
        if BLUE_AGENT_MODE == 0:
            blue_agent = DemoAutoAgent('blue', "blue_demo")
            print("[配置] 蓝方使用: Demo")
        elif BLUE_AGENT_MODE == 1:
            blue_agent = BTDemoAgentCopy('blue', "blue_bt_copy")
            print("[配置] 蓝方使用: 策略树copy")
        else:
            blue_agent = BTDemoAgent('blue', "blue_bt")
            print("[配置] 蓝方使用: 行为树（垂直躲避）")

        if config.is_single_instance:
            auto_engage_main(red_agent, blue_agent)
        else:
            auto_engage_main_multi(red_agent, blue_agent)
    elif config.current_config.battle_mode == config.ENGAGE_MODE_TEAMING:
        # 人机混合编组竞技模式
        red_agent = DemoTeamingAgent('red', "red_demo")
        blue_agent = DemoAutoAgent('blue', "blue_demo")
        # # 新智能体
        # red_agent = FlyTeamTeamingAgent('red', "red_demo")
        # blue_agent = FlyTeamAutoAgent('blue', "blue_demo")
        teaming_engage_main(red_agent, blue_agent)
