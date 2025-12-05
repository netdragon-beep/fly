import config
from env.env import auto_engage_main, teaming_engage_main
from env.multi_env import auto_engage_main_multi
from env.agent.demo.demo_auto_agent import DemoAutoAgent
from env.agent.demo.demo_teaming_agent import DemoTeamingAgent
from utilities.yxHttp import YxHttpRequest as yxHttp

if __name__ == '__main__':

    yxHttp.clear_room()
    if config.current_config.battle_mode == config.ENGAGE_MODE_AUTO:
        # 机器竞技模式
        red_agent = DemoAutoAgent('red', "red_demo")
        blue_agent = DemoAutoAgent('blue', "blue_demo")
        if config.is_single_instance:
            auto_engage_main(red_agent, blue_agent)
        else:
            auto_engage_main_multi(red_agent, blue_agent)
    elif config.current_config.battle_mode == config.ENGAGE_MODE_TEAMING:
        # 人机混合编组竞技模式
        red_agent = DemoTeamingAgent('red', "red_demo")
        blue_agent = DemoAutoAgent('blue', "blue_demo")
        teaming_engage_main(red_agent, blue_agent)