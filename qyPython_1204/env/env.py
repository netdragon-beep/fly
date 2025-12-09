import json, copy, time, config
from utilities.dataParser import PlatformData
from utilities.yxScriptTreeFunc import YxScriptTreeFunc
from utilities.yxWebsocket import YXWebSocketServer
from env.env_base import EnvBase

class Env(EnvBase):
    def __init__(self, red_agent=None, blue_agent=None):
        super().__init__()
        self.red_agent = red_agent
        self.blue_agent = blue_agent

        self.global_fun_url = config.server["globalFunUrl"]
        self.get_scenario_url = config.server["getScenarioUrl"]
        self.fun_server = None
        self.run_times = config.run_times

        if self.run_times > 0:
            self.current_run_time = self.run_times

        # 设置服务器连接
        self.fun_server = self.socket_manager.add_client(
            self.global_fun_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )

        self.funTool = YxScriptTreeFunc(self.fun_server)

        if config.current_config.battle_mode == config.ENGAGE_MODE_TEAMING:
            self.my_ws_server = YXWebSocketServer(host='0.0.0.0', port=9003)
            self.my_ws_server.funTool = self.funTool
            self.my_ws_server.start()

        self.web_server = self.socket_manager.add_client(
            self.get_scenario_url,
            on_open=self.on_scenario_open,
            on_message=self.on_scenario_message,
            on_error=self.on_scenario_error,
            on_close=self.on_scenario_close
        )

    def on_scenario_message(self, ws, message):
        data = json.loads(message)
        if 'fun' in data:
            fun = data['fun']
            if fun == 'scenarioContent':
                if 'platform' in data:
                    platforms = data['platform']
                    if platforms is not None:
                        self.scenario_content = platforms
                    for key in platforms:
                        platform = platforms[key]
                        self.funTool.sim_data.add_or_update_platform(platform)
            elif fun == 'simulationContent':
                self.funTool.reset()
                if 'platform' in data:
                    platforms = data['platform']
                    if platforms is not None:
                        self.scenario_content = platforms
                    for key in platforms:
                        platform = platforms[key]
                        self.funTool.sim_data.add_or_update_platform(platform)
                self.funTool.sim_start = True
                tdata = {'fun': 'SimulationStarting'}
                self.funTool.sim_data.update_header(tdata)
            elif fun == 'node':
                if 'wsAddress' in data:
                    self.wsAddress = data['wsAddress']
                    print(self.wsAddress)
                    if config.is_debug:
                        self.wsAddress = config.debug_url
                    self.scenario_ws = self.socket_manager.add_client(
                        self.wsAddress,
                        on_open=self.on_scenario_open,
                        on_message=self.on_scenario_message,
                        on_error=self.on_scenario_error,
                        on_close=self.on_scenario_close
                    )
                    self.funTool.set_scenario_ws(self.scenario_ws)
            elif fun == 'brokenPlatform':
                if 'list' in data:
                    tlist = data['list']
                    if len(tlist) > 0:
                        for item in tlist:
                            if item['moverType'] != '':
                                if item['moverType'] != 'WSF_GUIDED_MOVER':
                                    self.funTool.sim_data.broken_platform_info[item['side']].append(item['name'])
            elif fun == 'PlatformAdded':
                new_platform = PlatformData(data)
                if not isinstance(new_platform, dict):
                    new_platform = new_platform.to_dict(True)
                self.funTool.sim_data.add_or_update_platform(new_platform)
            elif fun == 'MoverUpdated':
                self.funTool.sim_data.add_or_update_platform(data)
            elif fun == 'PlatformDeleted' or fun == 'PlatformBroken':
                self.funTool.sim_data.remove_platform(data)
            elif fun == 'SensorTrackUpdated':
                self.funTool.sim_data.add_or_update_track(data)
            elif fun == 'SensorTrackDropped':
                self.funTool.sim_data.remove_track(data)
            elif fun == 'WeaponFired':
                self.funTool.sim_data.update_weapon_quantity(data)
            elif fun == 'AdvanceTime' or fun == 'SimulationStarting' or fun == 'SimulationComplete' or fun == 'simControl':
                self.funTool.sim_data.update_header(data)
                if fun == 'SimulationStarting':
                    self.funTool.sim_start = True
                    print('开始推演')
                elif fun == 'simControl':
                    if data['option'] == 'play':
                        self.funTool.sim_start = True
                elif fun == 'SimulationComplete':
                    self.funTool.sim_start = False
                    self.funTool.reset()



    def step(self):
        raw_observation = copy.deepcopy(self.funTool.get_sim_data())
        self.compute_center_time(raw_observation)

        red_cmds = []
        blue_cmds = []

        if self.red_agent is not None:
            red_cmds = self.red_agent.get_cmds(raw_observation['side_list'][0])
        if self.blue_agent is not None:
            blue_cmds = self.blue_agent.get_cmds(raw_observation['side_list'][1])

        # 3. 解析指令并调用YxScriptTreeFunc的对应方法
        for cmd in red_cmds + blue_cmds:
            try:
                self.funTool.send_str_ws(cmd['json_data'])
            except Exception as e:
                print(f"指令执行失败:{e}")
        return self.get_done()


def auto_engage_main(red_agent, blue_agent):
    env = Env(red_agent, blue_agent)
    while True:
        done = [False, False, False]
        start_once = True
        while not done[0]:
            # try:
                if env.funTool.s_ws is not None:
                    if not env.funTool.sim_start:
                        if start_once:
                            env.funTool.sim_control('edit')
                            time.sleep(2)
                            start_once = False
                            env.funTool.set_simulation_input(0.1)
                            env.funTool.sim_control('play', 'frameStepped')
                        else:
                            time.sleep(0.02)
                    else:
                        if env.funTool.sim_data.frame == 0:
                            done = env.step()
                            env.funTool.sim_data.frame = config.frame_num
                            env.funTool.sim_control('step')
                            env.wait_next_step = False
                        else:
                            if not env.wait_next_step:
                                env.wait_next_step = True
                                env.force_step_time = time.time()
                            else:
                                current_time = time.time() - env.force_step_time
                                if current_time >= 1:
                                    env.force_step_time = time.time()
                                    env.funTool.sim_data.frame = 0
                                    print('force step')
                        time.sleep(0.02)
            # except Exception as e:
            #     print(f'执行auto_engage_main时报错：{e}')
        print(f"推演结束:{done[0]} 红方胜利:{done[1]} 蓝方胜利:{done[2]}")
        env.funTool.sim_control('edit')
        if env.run_times <= 0:
            if config.is_print_debug:
                print(f"运行结束,即将开始下一轮")
        else:
            env.current_run_time -= 1
            if env.current_run_time == 0:
                if config.is_print_debug:
                    print(f"运行结束：全部结束")
                return
            else:
                if config.is_print_debug:
                    print(f"运行结束,剩余{env.run_times}轮,{config.run_delta_time}秒后开始下一轮")
        time.sleep(config.run_delta_time)


def teaming_engage_main(red_agent, blue_agent):
    last_time = 0
    env = Env(red_agent, blue_agent)
    done = [False, False, False]
    step_time = config.current_config.step_time
    while not done[0]:
        try:
            if env.funTool.s_ws is not None:
                if env.funTool.sim_start:
                    current_time = env.funTool.sim_data.header.sim_time
                    if current_time - last_time >= step_time:
                        last_time = current_time
                        done = env.step()
            time.sleep(0.02)
        except Exception as e:
            print(e)
    print(f"推演结束:{done[0]} 红方胜利:{done[1]} 蓝方胜利:{done[2]}")
    env.funTool.sim_control('edit')