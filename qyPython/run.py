import config
from env.env import auto_engage_main, teaming_engage_main
from env.multi_env import auto_engage_main_multi
from env.agent.demo.demo_auto_agent import DemoAutoAgent
from env.agent.demo.demo_teaming_agent import DemoTeamingAgent
from env.agent.策略树.Behavior_Tree_auto import BTDemoAgent  # 红方：V2动态NEZ版本
from env.agent.策略树_backup.Behavior_Tree_auto import BTDemoAgent as BTDemoAgentBackup  # 蓝方：备份版本
# from env.agent.test1.test1_auto_agent import FlyTeamAutoAgent
# from env.agent.test1.test1_teaming_agent import FlyTeamTeamingAgent
from utilities.yxHttp import YxHttpRequest as yxHttp

if __name__ == '__main__':

    yxHttp.clear_room()
    if config.current_config.battle_mode == config.ENGAGE_MODE_AUTO:
        # 机器竞技模式
        # 红方使用V2动态NEZ版本，蓝方使用备份版本（对比测试）
        red_agent = BTDemoAgent('red', "red_v2_nez")
        blue_agent = BTDemoAgentBackup('blue', "blue_backup")
        # # 新智能体
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