import json, copy, time, config
from utilities.dataParser import PlatformData
from utilities.yxScriptTreeFunc import YxScriptTreeFunc
from utilities.yxWebsocket import YXWebSocketClientManager
import multiprocessing as mp
from utilities.yxHttp import YxHttpRequest as yxHttp
from utilities.yxWebsocket_local import YXWebSocketServerMain as yxWS
from env.env_base import EnvBase


class MultiEnv(EnvBase):
    def __init__(self, room_id=None, ws_url=None, is_child=False):
        # 创建管理器
        super().__init__()
        self.socket_manager = YXWebSocketClientManager()
        # 初始化类变量
        self.wsAddress = ''
        self.get_scenario_url = ws_url
        self.yxWS = None
        self.room_id = room_id
        self.fun_ws = None
        self.is_child = is_child
        self.do_next_step = True
        self.cmds = None
        self.done = [False, False, False]
        self.run_times = config.run_times
        if self.run_times > 0:
            self.current_run_time = self.run_times

        if self.is_child:
            self.yxWS = self.socket_manager.add_client(
                'ws://127.0.0.1:9001',
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
        # 设置服务器连接
        self.funTool = YxScriptTreeFunc(self.fun_ws, room_id)
        self.web_server = self.socket_manager.add_client(
            self.get_scenario_url,
            on_open=self.on_scenario_open,
            on_message=self.on_scenario_message,
            on_error=self.on_scenario_error,
            on_close=self.on_scenario_close
        )

        if self.web_server is not None:
            self.funTool.set_scenario_ws(self.web_server)
        else:
            print("推演服务器连接错误！！！")

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
                    print(f"实例{self.room_id}开始推演")
                elif fun == 'simControl':
                    if data['option'] == 'play':
                        self.funTool.sim_start = True
                elif fun == 'SimulationComplete':
                    self.funTool.sim_start = False
                    self.funTool.reset()
                    print(f"实例{self.room_id}结束推演")

    def on_message(self, ws, message):
        data = json.loads(message)
        if 'fun' in data:
            fun = data['fun']
            if fun == 'doNextStep':
                self.cmds = data['cmds']
                self.do_next_step = True

    def send_msg_main(self, data) -> bool:
        if self.yxWS is None:
            print("未获取到ws，请重试")
            return False
        if self.room_id is not None:
            data['roomId'] = self.room_id
        json_data = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
        if config.is_print_debug:
            print(f'发送消息：{json_data}')
        return self.yxWS.send(json_data)

    def run(self):
        while True:
            self.done = [False, False, False]
            start_once = True
            while not self.done[0]:
                if self.funTool.s_ws is not None:
                    if not self.funTool.sim_start:
                        if start_once:
                            time.sleep(2)
                            start_once = False
                            self.funTool.set_simulation_input(0.1)
                            self.funTool.sim_control('play', 'frameStepped')
                        else:
                            time.sleep(0.02)
                    else:
                        if self.funTool.sim_data.frame == 0:
                            if not self.wait_next_step:
                                current_sim_data = self.funTool.sim_data.to_dict()
                                current_sim_data['room_id'] = self.room_id
                                msg = {
                                    "fun":"waitNextStep",
                                    "simData": current_sim_data
                                }
                                self.send_msg_main(msg)
                                self.wait_next_step = True
                                self.force_step_time = time.time()
                            elif self.wait_next_step and self.do_next_step:
                                self.force_step_time = time.time()
                                self.done = self.step()
                                self.funTool.sim_data.frame = config.frame_num
                                self.funTool.sim_control('step')
                                self.wait_next_step = False
                                self.do_next_step = False
                            else:
                                current_time = time.time() - self.force_step_time
                                if current_time >= 1:
                                    self.force_step_time = time.time()
                                    self.force_step()
                                pass
                        time.sleep(0.02)
                else:
                    print(f"实例{self.room_id}未初始化成功")
            print(f"实例{self.room_id}推演结束:{self.done[0]} 红方胜利:{self.done[1]} 蓝方胜利:{self.done[2]}")
            self.funTool.sim_control('edit')
            if self.run_times <= 0:
                if config.is_print_debug:
                    print(f"运行结束,即将开始下一轮")
            else:
                self.current_run_time -= 1
                if self.current_run_time == 0:
                    msg = {"fun": "simulationCompleted"}
                    self.send_msg_main(msg)
                    return
                else:
                    if config.is_print_debug:
                        print(f"运行结束,剩余{self.run_times}轮,{config.run_delta_time}秒后开始下一轮")
            time.sleep(config.run_delta_time)

    def force_step(self):
        self.wait_next_step = False
        print(f'实例{self.room_id}强制更新')

    def step(self):
        # 3. 解析指令并调用YxScriptTreeFunc的对应方法
        if self.cmds is not None:
            for cmd in self.cmds:
                try:
                    cmd = cmd["json_data"]
                    self.funTool.send_str_ws(cmd)
                except Exception as e:
                    print(f"指令执行失败：{e}")
            self.cmds = {}
        return self.get_done()


def env_process(room_id, ws_url):
    try:
        if room_id is None:
            print("多实例想定启动时，必须传入实例id和推演服务器地址")
        env = MultiEnv(room_id, ws_url, True)
        env.run()

    except Exception as e:
        print(f"实例{room_id}结束,原因：{str(e)}")

def auto_engage_main_multi(red_agent, blue_agent):
    ws = yxWS(host='0.0.0.0', port=9001)
    ws.start()
    process_list = []
    print("开始执行主进程...")
    yxHttp.clear_room()
    time.sleep(0.5)
    yxHttp.generate_room()
    can_use_rooms = yxHttp.get_and_check_room_list()
    current_room_num = len(can_use_rooms)
    if current_room_num > 0:
        while current_room_num < config.generate_task_num:
            print(f'当前可用实例数{current_room_num},目标实例数{config.generate_task_num},等待实例生成中...')
            time.sleep(0.1)
            can_use_rooms = yxHttp.get_and_check_room_list()
            current_room_num = len(can_use_rooms)
        print("实例生成完成")
        current_step_num = 0
        ws.process_num = len(can_use_rooms)
        for room in can_use_rooms:
            p = mp.Process(
                target=env_process,
                args=(
                    room['roomId'],
                    room['wsAddress']
                ),
                daemon=True
            )
            process_list.append(p)
            p.start()
            print(f"开始进程:进程id-{p.pid} 实例id-{room['roomId']} 推演服务器地址-{room['wsAddress']}")
        try:
            while ws.process_num > 0:
                current_step_num = 0
                if not config.is_async_step:
                    for k, b in ws.wait_dict.items():
                        if b:
                            current_step_num += 1
                if current_step_num == ws.process_num or config.is_async_step:
                    if len(ws.sim_data_dict) > 0:
                        sim_datas = {}
                        ids = []
                        with ws.dict_lock:
                            sim_datas = copy.deepcopy(ws.sim_data_dict)
                        for tid, raw_observation in sim_datas.items():
                            ids.append(tid)
                            red_cmds = []
                            blue_cmds = []
                            if raw_observation is not None and 'side_list' in raw_observation:
                                if red_agent is not None:
                                    red_cmds = red_agent.get_cmds(raw_observation['side_list'][0])
                                if blue_agent is not None:
                                    blue_cmds = blue_agent.get_cmds(raw_observation['side_list'][1])
                                msg = {
                                    "fun": "doNextStep",
                                    "cmds": red_cmds + blue_cmds
                                }
                                ws.send_message(tid, msg)
                            else:
                                time.sleep(0.02)
                        with ws.dict_lock:
                            for tid in ids:
                                ws.sim_data_dict.pop(tid)
                        if config.is_print_debug:
                            if len(ws.sim_data_dict) > 0:
                                print(f'存在数据：{ws.sim_data_dict}')
                    else:
                        if config.is_print_debug:
                            print("未获取到想定数据")
                        time.sleep(0.02)
                else:
                    time.sleep(0.1)
        except Exception as e:
            print(f"退出,{str(e)}")

        for p in process_list:
            p.join()
    else:
        print('实例生成失败，请重试运行...')