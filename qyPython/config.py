from collections import namedtuple

user_id = 97 # 用户id（默认不修改）

# 想定(task_id)：
## 2904 -> 机机竞技
## 2911 -> 人机混合编组（测试智能体）
## 3656 -> 人机混合编组（测试操纵杆）

# 定义竞技模式常量
ENGAGE_MODE_AUTO = "机机竞技"
ENGAGE_MODE_TEAMING = "人机混合编组竞技"

Config = namedtuple("Config", field_names=[
    "task_id",                      # 想定id
    "battle_mode",                  # 竞技模式：机机竞技/人机混合编组竞技
    "is_single_instance",           # 机机竞技：是否单例运行
    "is_async_step",                # 机机竞技：是否异步Step获取态势（非单例运行时有效）
    "generate_task_num",            # 机机竞技：生成实例数量（非单例运行时有效）
    "run_times",                    # 机机竞技：实例运行轮数，<=0时无限跑
    "frame_num",                    # 机机竞技：获取态势数据的间隔帧数
    "run_delta_time",               # 机机竞技：实例结束时等待下一轮的间隔时间（秒）
    "step_time"                     # 人机混合编组竞技：每个step的时间间隔（秒）
],defaults=[None, None, None, None, None, 1, None, None, None])

# 机机竞技模式配置
AUTO_ENGAGE_CONFIG = Config(
    task_id=2904,
    battle_mode=ENGAGE_MODE_AUTO,
    is_single_instance=True,
    is_async_step=True,
    generate_task_num=2,
    run_times=3,
    frame_num=5,
    run_delta_time=5
)

# 人机混合编组竞技模式配置
TEAMING_ENGAGE_CONFIG = Config(
    task_id=2911, #3656
    battle_mode=ENGAGE_MODE_TEAMING,
    step_time=0.5
)

# 获取当前配置
current_config = AUTO_ENGAGE_CONFIG # {AUTO_ENGAGE_CONFIG, TEAMING_ENGAGE_CONFIG}

task_id = current_config.task_id
is_single_instance = current_config.is_single_instance
generate_task_num = current_config.generate_task_num
run_times = current_config.run_times
frame_num = current_config.frame_num
run_delta_time = current_config.run_delta_time
is_async_step = current_config.is_async_step


# 其他配置
server = {
    # ---------------本地---------------

    "ip": "127.0.0.1",
    "port": 9033,
    "war": "ArkSimServer",
    "globalFunUrl": "ws://127.0.0.1:9033/ArkSimServer/func/" + str(user_id),
    "getScenarioUrl": "ws://127.0.0.1:9033/ArkSimServer/bg/" + str(user_id) + "/" + str(task_id)

}

request_urls = {
    "room_list": {
        "url": f"http://{server['ip']}:{server['port']}/{server['war']}/taskRoom/roomList",
        "params": {
            "taskId": task_id,
            "pageSize": 1000,
        }
    },
    "generate_room": {
        "url": f"http://{server['ip']}:{server['port']}/{server['war']}/taskRoom/generateRoom",
        "params": {
            'taskId': task_id,
            'runTimes': generate_task_num,
            'operate': 'generate'
        }
    },
    "edit_room": {
        "url": f"http://{server['ip']}:{server['port']}/{server['war']}/taskRoom/generateRoom",
        "params": {
            'taskId': task_id,
            'operate': 'edit'
        }
    },
    "clear_room": {
        "url": f"http://{server['ip']}:{server['port']}/{server['war']}/taskRoom/generateRoom",
        "params": {
            'taskId': task_id,
            'operate': 'delete'
        }
    }
}

# 调试地址
debug_url = 'ws://' + '192.168.1.105:60272'

# is_debug = True
is_debug = False

# is_print_debug = True
is_print_debug = False

send_to_ws = True
# send_to_ws = False
