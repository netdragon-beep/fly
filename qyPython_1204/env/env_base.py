import config, math, copy, json
from utilities.yxWebsocket import YXWebSocketClientManager
from utilities.yxHttp import YxHttpRequest as yxHttp

class EnvBase:
    def __init__(self):
        # 创建管理器
        self.socket_manager = YXWebSocketClientManager()

        # 初始化类变量
        self.wsAddress = ''
        self.get_scenario_url = None
        self.ws_url = None
        self.scenario_content = None
        self.web_server = None
        self.scenario_ws = None
        self.funTool = None
        self.wait_next_step = False
        self.force_step_time = None


        self.red_center_time = 0
        self.blue_center_time = 0

        self.battlefield = {
            'min_lon': 0,  # 我方初始竖线经度
            'max_lon': 0,  # 敌方初始竖线经度
            'min_lat': 0,  # 战场最小纬度（下边界）
            'max_lat': 0,  # 战场最大纬度（上边界）
        }

        task_list = yxHttp.get_task_list()
        for task in task_list:
            if task['taskId'] == config.task_id:
                if 'positions' in task:
                    positions = task['positions']
                    positions.replace('\\', '')
                    position_list = json.loads(positions)
                    min_lat, min_lon, max_lat, max_lon = float(position_list[0]['latitude']), float(position_list[0]['longitude']), float(position_list[1]['latitude']), float(position_list[1]['longitude'])

                    if min_lon > max_lon:
                        min_lon, max_lon = max_lon, min_lon
                    if min_lat > max_lat:
                        min_lat, max_lat = max_lat, min_lat
                    self.battlefield = {
                        'min_lon': min_lon,  # 我方初始竖线经度
                        'max_lon': max_lon,  # 敌方初始竖线经度
                        'min_lat': min_lat,  # 战场最小纬度（下边界）
                        'max_lat': max_lat,  # 战场最大纬度（上边界）
                    }



    def set_or_update_scenario_content(self, data):
        if 'platform' in data:
            platforms = data['platform']
            if platforms is not None:
                self.scenario_content = platforms

    def set_batllefield(self, battlefield):
        self.battlefield = battlefield

    def on_scenario_error(self, ws, error):
        print(error)

    def on_scenario_close(self, ws, close_status_code, close_msg):
        print(f'scenario closed by {close_status_code} : {close_msg}')

    def on_scenario_open(self, ws):
        if config.is_print_debug:
            print(f'===================connected to {ws.url}===================')

    def on_message(self, ws, message):
        if config.is_print_debug:
            print(message)

    def on_error(self, ws, error):
        if config.is_print_debug:
            print(error)

    def on_close(self, ws, close_status_code, close_msg):
        if config.is_print_debug:
            print("### closed ###")

    def on_open(self, ws):
        if config.is_print_debug:
            print("Opened connection")

    def compute_center_time(self, raw_observation):
        # 地球半径（公里）
        EARTH_RADIUS_KM = 6371.0

        def is_in_circle(lat1, lon1, lat2, lon2, radius_km):
            """计算两点间距离并判断是否在指定半径内"""
            # 将角度转换为弧度
            lat1_rad = math.radians(lat1)
            lon1_rad = math.radians(lon1)
            lat2_rad = math.radians(lat2)
            lon2_rad = math.radians(lon2)

            # 计算经纬度差值
            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad

            # 使用Haversine公式计算大圆距离
            a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            distance = EARTH_RADIUS_KM * c

            return distance <= radius_km

        center_lon = (self.battlefield['min_lon'] + self.battlefield['max_lon']) / 2
        center_lat = (self.battlefield['min_lat'] + self.battlefield['max_lat']) / 2
        for side_item in raw_observation['side_list']:
            for platform_item in side_item['platform_list']:
                if platform_item['type'] == "有人机":
                    lat = platform_item['latitude']
                    lon = platform_item['longitude']

                    # 判断是否在圆形区域内
                    if is_in_circle(center_lat, center_lon, lat, lon, 5):
                        if side_item['side'] == 'red':
                            self.red_center_time += 1
                        elif side_item['side'] == 'blue':
                            self.blue_center_time += 1
                        break

    def get_done(self):
        raw_observation = copy.deepcopy(self.funTool.get_sim_data())
        missile_num = 0
        for side_item in raw_observation['side_list']:
            for platform_item in side_item['platform_list']:
                for weapon in platform_item['weapons']:
                    missile_num += weapon['quantity']
            for track_item in side_item['track_list']:
                if track_item['platform_entity_type'] == "导弹":
                    missile_num += 1

        # 终止标志，红方胜利，蓝方胜利
        done = [False, False, False]
        if raw_observation['header']['sim_time'] > 60:
            red_uav = [item for item in raw_observation['side_list'][0]['platform_list'] if
                       item['type'] == "无人机"]
            red_manned = [item for item in raw_observation['side_list'][0]['platform_list'] if
                          item['type'] == "有人机"]
            blue_uav = [item for item in raw_observation['side_list'][1]['platform_list'] if
                        item['type'] == "无人机"]
            blue_manned = [item for item in raw_observation['side_list'][1]['platform_list'] if
                           item['type'] == "有人机"]

            red_uav_num = len(red_uav)
            red_manned_num = len(red_manned)
            blue_uav_num = len(blue_uav)
            blue_manned_num = len(blue_manned)

            if red_manned_num == 0:
                done[0] = True
                done[2] = True
                return done
            elif blue_manned_num == 0:
                done[0] = True
                done[1] = True
                return done

            if missile_num == 0 or raw_observation['header']['sim_time'] > 15 * 60 - 1:
                done[0] = True
                if red_uav_num != blue_uav_num:
                    done[1] = red_uav_num > blue_uav_num
                    done[2] = blue_uav_num > red_uav_num
                    return done
                else:
                    done[1] = self.red_center_time > self.blue_center_time
                    done[2] = self.blue_center_time > self.red_center_time
                    return done

        return done
