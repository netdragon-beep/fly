import sys
import os
import json
import config


def load_config():
    # 判断程序是否已打包
    if getattr(sys, 'frozen', False):
        # 如果是打包后的程序，sys.executable 是 exe 文件的路径
        exe_dir = os.path.dirname(sys.executable)
        # 配置文件位于 exe 文件同目录下
        config_path = os.path.join(exe_dir, 'config.json') # 请将 'config.json' 替换为你的配置文件名
    else:
        # 开发环境下的配置文件路径
        config_path = 'config.json'
    # 尝试读取外部配置文件
    config_data = {}
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f) # 如果是其他格式如ini，使用相应解析方法
        print(f"成功加载外部配置文件: {config_path}")
        if config_data is not None:
            for key, value in config_data.items():
                if key == "user_id":
                    config.user_id = value
                elif key == "task_id":
                    config.task_id = value
                elif key == "end_time":
                    config.end_time = value
                elif key == "run_times":
                    config.run_times = value
                elif key == "run_delta_time":
                    config.run_delta_time = value
                elif key == "frame_num":
                    config.frame_num = value
                elif key == "is_single_instance":
                    config.is_single_instance = value
                elif key == "is_async_step":
                    config.is_async_step = value
                elif key == "generate_task_num":
                    config.generate_task_num = value
                elif key == "server":
                    config.server = value
                    config.server['globalFunUrl'] += str(config.user_id)
                    config.server['getScenarioUrl'] += str(config.user_id) + "/" + str(config.task_id)
                elif key == "request_urls":
                    config.request_urls = value
                    url = f"http://{config.server['ip']}:{config.server['port']}/{config.server['war']}/"
                    room_list = config.request_urls['room_list']
                    generate_room = config.request_urls['generate_room']
                    edit_room = config.request_urls['edit_room']
                    clear_room = config.request_urls['clear_room']
                    config.request_urls['room_list'] = {
                        'url': url + room_list,
                        'params':{
                            'taskId': config.task_id,
                            'pageSize': 1000
                        }
                    }
                    config.request_urls['generate_room'] = {
                        'url': url + generate_room,
                        'params':{
                            'taskId': config.task_id,
                            'runTimes': config.generate_task_num,
                            'operate': 'generate'
                        }
                    }
                    config.request_urls['edit_room'] = {
                        'url': url + edit_room,
                        'params': {
                            'taskId': config.task_id,
                            'operate': 'edit'
                        }
                    }
                    config.request_urls['clear_room'] = {
                        'url': url + clear_room,
                        'params': {
                            'taskId': config.task_id,
                            'operate': 'delete'
                        }
                    }
                elif key == "is_debug":
                    config.is_debug = value
                elif key == "is_print_debug":
                    config.is_print_debug = value
                elif key == "send_to_ws":
                    config.send_to_ws = value
    except FileNotFoundError:
        # 如果配置文件不存在，可以在这里创建一份默认配置，或者抛出错误信息
        print(f"未找到外部配置文件 {config_path}，请确保其与程序在同一目录。程序将使用内置默认配置或退出。")
        # 可以选择创建一个默认的配置文件
        default_config = {
            "user_id": 97,
            "task_id": 2904,
            "end_time": 3600,
            "run_times": 3,
            "run_delta_time": 2,
            "frame_num": 5,
            "is_single_instance": False,
            "is_async_step": False,
            "generate_task_num": 3,
            "server": {
                "ip": "127.0.0.1",
                "port": 9033,
                "war": "ArkSimServer",
                "globalFunUrl": "ws://127.0.0.1:9033/ArkSimServer/func/",
                "getScenarioUrl": "ws://127.0.0.1:9033/ArkSimServer/bg/"
            },
            "request_urls": {
                "room_list": "taskRoom/roomList",
                "generate_room": "taskRoom/generateRoom",
                "edit_room": "taskRoom/generateRoom",
                "clear_room": "taskRoom/generateRoom"
            },
            "debug_url": "",
            "is_debug": False,
            "is_print_debug": False,
            "send_to_ws": True,
            "battle_task_id": 2904
        }
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
            print(f"已在 {config_path} 创建默认配置文件。")
        except Exception as e:
            print(f"创建默认配置文件失败: {e}")
    except Exception as e:
        print(f"读取配置文件时发生错误: {e}")