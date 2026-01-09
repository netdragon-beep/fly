import requests
import config
import json


class YxHttpRequest:
    def __init__(self):
        pass

    @staticmethod
    def get_room_list():
        response = requests.get(config.request_urls['room_list']['url'], params=config.request_urls['room_list']['params'])
        if response.status_code == 200:
            data = response.json()['data']
            print(f"当前任务数：{data['total']}")


    @staticmethod
    def generate_room():
        print(f"正在生成{config.generate_task_num}个新实例，请稍候...")
        response = requests.get(config.request_urls['generate_room']['url'], params=config.request_urls['generate_room']['params'])
        if response.status_code == 200:
            print(f"调用生成实例成功，{response.text}")
        if config.is_print_debug:
            print(response.url)

    @staticmethod
    def get_and_check_room_list()-> list:
        can_use_rooms = []
        try:
            response = requests.get(config.request_urls['room_list']['url'], params=config.request_urls['room_list']['params'])
            if response.status_code == 200:
                data = response.json()['data']
                room_list = data['list']
                if len(room_list) > 0:
                    for item in room_list:
                        print(str(item['roomId']) + '：' + item['wsAddress'] + '  ' + str(item['status']))
                        room = {'roomId': item['roomId'], 'wsAddress': item['wsAddress']}
                        if item['status'] == 0:
                            can_use_rooms.append(room)
            if config.is_print_debug:
                print(response.url)
        except Exception as e:
            print(f'生成实例失败{e}')
        return can_use_rooms

    @staticmethod
    def edit_room():
        response = requests.get(config.request_urls['edit_room']['url'], params=config.request_urls['edit_room']['params'])
        if response.status_code == 200:
            print("停止所有实例成功")
        else:
            print(f"停止所有实例失败：{response.status_code}")

    @staticmethod
    def clear_room():
        response = requests.get(config.request_urls['clear_room']['url'], params=config.request_urls['clear_room']['params'])
        if response.status_code == 200:
            print(f"清除所有实例成功")
        else:
            print(f"清理所有实例失败：{response.status_code}")

    @staticmethod
    def get_task_list():
        url = f"http://{config.server['ip']}:{config.server['port']}/{config.server['war']}/task/taskList"
        response = requests.post(url, params={'userId':config.user_id})
        if response.status_code == 200:
            json = response.json()
            return json['data']
        return {}

    @staticmethod
    def get_task_battlefield():
        battlefield = {
            'min_lon': None,  # 我方初始竖线经度
            'max_lon': None,  # 敌方初始竖线经度
            'min_lat': None,  # 战场最小纬度（下边界）
            'max_lat': None,  # 战场最大纬度（上边界）
        }
        task_list = YxHttpRequest.get_task_list()
        for task in task_list:
            if task['taskId'] == config.task_id:
                if 'positions' in task:
                    positions = task['positions']
                    positions.replace('\\', '')
                    position_list = json.loads(positions)
                    min_lat, min_lon, max_lat, max_lon = float(position_list[0]['latitude']), float(
                        position_list[0]['longitude']), float(position_list[1]['latitude']), float(
                        position_list[1]['longitude'])

                    if min_lon > max_lon:
                        min_lon, max_lon = max_lon, min_lon
                    if min_lat > max_lat:
                        min_lat, max_lat = max_lat, min_lat
                    battlefield = {
                        'min_lon': min_lon,  # 我方初始竖线经度
                        'max_lon': max_lon,  # 敌方初始竖线经度
                        'min_lat': min_lat,  # 战场最小纬度（下边界）
                        'max_lat': max_lat,  # 战场最大纬度（上边界）
                    }
        return battlefield