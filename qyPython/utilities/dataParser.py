import math
import time

import config


class Mover:#机动器属性
    def __init__(self, data):
        self.id = None
        self.name = None

        if data == '':
            self.id = 0
            self.name = ''
        else:
            if isinstance(data, dict):
                self.update(data.items())
            elif data is not None:
                self.update(data)

    def update(self, data):
        for key, value in data:
            if key == 'id' or key == 'moverId':
                self.id : int = value
            elif key == 'name' or key == 'moverName':
                self.name : str = value

    def to_dict(self):
        return self.__dict__

class Weapon:#武器属性
    def __init__(self, data):
        self.id = None
        self.name = None
        self.launched_platform_type = None
        self.type = None
        self.quantity = None

        if isinstance(data, dict):
            self.update(data.items())
        elif data is not None:
            self.update(data)

    def update(self, data):
        for key, value in data:
            if key == 'id':
                self.id = value
            elif key == 'name':
                self.name = value
            elif key == 'launched_platform_type':
                self.launched_platform_type = value
            elif key == 'type':
                self.type = value
            elif key == 'quantity':
                self.quantity = value

    def to_dict(self):
        return self.__dict__

class Sensor:#传感器属性
    def __init__(self, data):
        self.id = None
        self.name = None
        self.type = None
        self.turned_on = None
        self.az_max = None
        self.az_min = None
        self.el_max = None
        self.el_min = None
        self.max_range = None

        self.update(data)

    def update(self, data):
        if isinstance(data, dict):
            data = data.items()
        for key, value in data:
            if key == 'id':
                self.id = value
            elif key == 'name':
                self.name = value
            elif key == 'type':
                self.type = value
            elif key == 'initiallyTurnedOn':
                self.turned_on = value
            elif key == 'modeList':
                range_limits = []
                field_of_view = {}
                antenna = None
                try:
                    if 'default' in value['mode']['enum']:
                        beams = value['mode']['enum']['default']['beams']
                    else:
                        beams = value['mode']['enum']['']['beams']
                    if isinstance(beams, dict):
                        antenna = beams['1']['xmtrRcvr']['antenna']
                    elif isinstance(beams, list):
                        antenna = beams[0]['xmtrRcvr']['antenna']
                    if antenna is not None:
                        range_limits = antenna['rangeLimits']
                        field_of_view = antenna['field_of_view']
                except Exception as e:
                    print(f'设置field_of_view失败！原因：{e}')
                if field_of_view and isinstance(field_of_view, dict):
                    if 'maxAz' in field_of_view:
                        self.az_max = field_of_view['maxAz']
                        self.az_min = field_of_view['minAz']
                        self.el_max = field_of_view['maxEl']
                        self.el_min = field_of_view['minEl']
                    elif 'value' in field_of_view and 'enum' in field_of_view:
                        value = field_of_view['value']
                        if value is not None:
                            if 'rectangular' == value:
                                self.az_max = field_of_view['enum'][value]['azimuth_field_of_view']['maxAzFOV']
                                self.az_min = field_of_view['enum'][value]['azimuth_field_of_view']['minAzFOV']
                                self.el_max = field_of_view['enum'][value]['elevation_field_of_view']['maxElFOV']
                                self.el_min = field_of_view['enum'][value]['elevation_field_of_view']['minElFOV']
                if len(range_limits) > 0 and isinstance(range_limits, list):
                    self.max_range = range_limits[1]
            elif key == 'az_max':
                self.az_max = value
            elif key == 'az_min':
                self.az_min = value
            elif key == 'el_max':
                self.el_max = value
            elif key == 'el_min':
                self.el_min = value
            elif key == 'max_range':
                self.max_range = value

    def to_dict(self):
        return self.__dict__

class Track:#目标属性
    def __init__(self, data, track_platform):
        self.target_id = None
        self.target_name = None
        self.platform_entity_type = None    # 目标推测类型
        self.platform_entity_side = track_platform.side   # 阵营（例如 "BLUE" 或 "RED"）
        self.longitude = None
        self.latitude = None
        self.altitude = None
        self.speed = track_platform.speed
        self.v_x = track_platform.velocity_x
        self.v_y = track_platform.velocity_y
        self.v_z = track_platform.velocity_z
        self.heading = track_platform.heading    # 偏航角（-π ~ π）
        self.pitch = track_platform.pitch  # 俯仰角（-π/2 ~ π/2）
        self.roll = track_platform.roll   # 横滚角（-π ~ π）
        self.is_fired_num : int = 0 # Track目标被我方攻击的数量
        self.engage_target = None  # 攻击的目标（仅我方导弹有该属性，非导弹为None ）

        if isinstance(data, dict):
            self.update(data.items())
        elif data is not None:
            self.update(data)

    def update(self, data):
        for key, value in data:
            if key == 'targetPlatformId':
                self.target_id = value
            elif key == 'targetPlatformName':
                self.target_name = value
            elif key == 'targetPlatformType':
                self.platform_entity_type = value
            elif key == 'platform_entity_side':
                self.platform_entity_side = value
            elif key == 'lon':
                self.longitude = value
            elif key == 'lat':
                self.latitude = value
            elif key == 'alt':
                self.altitude = value
            elif key == 'is_fired_num':
                self.is_fired_num = value
            elif key == 'engage_target':
                self.engage_target = value
            elif key == 'track':
                for tkey, tvalue in value.items():
                    if tkey == 'lon':
                        self.longitude = tvalue
                    elif tkey == 'lat':
                        self.latitude = tvalue
                    elif tkey == 'alt':
                        self.altitude = tvalue

    def to_dict(self):
        return self.__dict__

class PathComputer:#气动模型参数
    def __init__(self, data):
        self.maximum_flight_path_angle : float = 85.5
        self.maximum_linear_acceleration : float = 58.8399
        self.maximum_radial_acceleration : float = 57.182080247115
        self.turn_g_limit : float = 56.3349
        self.maximumTurn : float = 5.6549

    def to_dict(self):
        return self.__dict__

class PlatformData:#平台属性
    def __init__(self, data):
        self.id = None
        self.side = None #red / blue
        self.type = None
        self.speed = None
        self.name = None
        self.longitude = None
        self.latitude = None
        self.altitude = None
        self.heading = None
        self.pitch = None
        self.roll = None
        self.velocity_x = None
        self.velocity_y = None
        self.velocity_z = None
        self.is_controlled = None
        self.mover = None
        self.sensors = {}
        self.weapons = {}
        self.track_list = None

        self.update(data)

    def update(self, data):
        if isinstance(data, dict):
            data = data.items()
        for key, value in data:
            if key == 'id':
                self.id = value
            elif key == 'side':
                self.side = value
            elif key == 'type':
                self.type = value
            elif key == 'speed':
                self.speed = value
            elif key == 'name':
                self.name = value
            elif key == 'longitude':
                self.longitude = value
            elif key == 'latitude':
                self.latitude = value
            elif key == 'altitude':
                self.altitude = value
            elif key == 'heading':
                self.heading = math.radians(value)
            elif key == 'pitch':
                self.pitch = math.radians(value)
            elif key == 'roll':
                self.roll = math.radians(value)
            elif key == 'velocityNED':
                self.velocity_x = value[0]
                self.velocity_y = value[1]
                self.velocity_z = value[2]
            elif key == 'is_controlled':
                self.is_controlled = value
            elif key == 'movers':
                if value is not None:
                    if len(value) > 0:
                        if self.mover is None:
                            mover = value['mover']
                            if mover is None:
                                self.mover = Mover(value['mover'])
                            else:
                                self.mover = Mover('')
                        else:
                            self.mover.update(data)
                    else:
                        self.mover = None
            elif key == 'mover':
                if self.mover is None:
                    self.mover = Mover(value)
                else:
                    self.mover.update(value)
            elif key == 'sensors':
                if value is not None:
                    if isinstance(value, dict):
                        for sensorName, sensor in value.items():
                            if sensorName in self.sensors:
                                self.sensors[sensorName].update(sensor)
                            else:
                                self.sensors[sensorName] = Sensor(sensor)
                    else:
                        for sensor in value:
                            sensorName = sensor['name']
                            if sensorName in self.weapons:
                                self.sensors[sensorName].update(sensor)
                            else:
                                self.sensors[sensorName] = Sensor(sensor)
            elif key == 'weapons':
                if value is not None:
                    if isinstance(value, dict):
                        for weapon_name, weapon in value.items():
                            if weapon_name in self.weapons:
                                self.weapons[weapon_name].update(weapon)
                            else:
                                self.weapons[weapon_name] = Weapon(weapon)
                    else:
                        for weapon in value:
                            weaponName = weapon['name']
                            if weaponName in self.weapons:
                                self.weapons[weaponName].update(weapon)
                            else:
                                self.weapons[weaponName] = Weapon(weapon)
            elif key == '#track_manager':
                if value is not None:
                    track_list = value['#track_manager']['trackList']
                    self.track_list = track_list

    def to_dict(self, need_track_list = False):
        dic = self.__dict__
        if self.mover:
            if not isinstance(self.mover, dict):
                dic['mover'] = self.mover.to_dict()
            else:
                dic['mover'] = self.mover
        if self.sensors:
            tsensors = []
            if isinstance(self.sensors, dict):
                for key, value in self.sensors.items():
                    tsensors.append(value.to_dict())
            else:
                for sensor in self.sensors:
                    tsensors.append(sensor)
            dic['sensors'] = tsensors
        if self.weapons:
            tweapons = []
            if isinstance(self.weapons, dict):
                for key, value in self.weapons.items():
                    tweapons.append(value.to_dict())
            else:
                for weapon in self.weapons:
                    tweapons.append(weapon)
            dic['weapons'] = tweapons
        if not need_track_list:
            if 'track_list' in dic:
                dic.pop('track_list')
        return dic

class SideListItem:
    def __init__(self, side):
        self.side = side
        self.platform_list = []
        self.track_list = []

    def to_dict(self):
        dic = self.__dict__
        pdic = []
        tdic = []
        if self.platform_list:
            for platform in self.platform_list:
                ptdic = platform.to_dict()
                pdic.append(ptdic)
        if self.track_list:
            for track in self.track_list:
                ttdic = track.to_dict()
                tdic.append(ttdic)
        dic['platform_list'] = pdic
        dic['track_list'] = tdic
        return dic

class Header:
    def __init__(self):
        self.sim_time : float = 0
        self.real_time : float = 0
        self.is_active : bool = False
        self.time_ratio : float = 1

    def update(self, data):
        for key, value in data.items():
            if key == 'sim_time':
                self.sim_time = value
            elif key == 'real_time':
                self.real_time = value
            elif key == 'is_active':
                self.is_active = value
            elif key == 'time_ratio':
                self.time_ratio = value

    def to_dict(self):
        dic = self.__dict__
        dic['real_time'] = time.time()
        return dic


class SimData:
    def __init__(self):
        self.header = Header()
        self.side_list = []
        self.platform_info = {}
        self.track_info = {'red': {}, "blue": {}}
        self.side_info = {'red': [], 'blue': []}
        self.broken_platform_info = {'red': [], 'blue': []}
        self.track_dict = {}
        self.id_name_dict = {}
        self.rate_list = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        self.current_rate = 1
        self.frame = 1
        self.user_side = 'red'

    def set_user_side(self, user_side):
        self.user_side = user_side

    def set_frame(self, frame):
        self.frame = frame - 1

    def update_header(self, data):
        fun = data['fun']
        update_data = {}
        if fun is not None:
            if fun == 'AdvanceTime':
                if self.header.sim_time != data['simTime']:
                    self.header.sim_time = data['simTime']
                    if self.frame > 0:
                        self.frame = self.frame - 1
                    # update_data['sim_time'] = data['simTime']
            elif fun == 'RealTime':
                update_data['real_time'] = data['realTime']
            elif fun == 'SimulationStarting':
                update_data['is_active'] = True
            elif fun == 'SimulationComplete':
                update_data['is_active'] = False
            elif fun == 'simControl':
                if 'rate' in data:
                    update_data['time_ratio'] = data['rate']
            self.header.update(update_data)

    def add_or_update_platform(self, data):
        if 'name' in data:
            platform_name = data['name']
            if platform_name in self.platform_info:
                self.platform_info[platform_name].update(data.items())
            else:
                platform = PlatformData(data.items())
                self.platform_info[platform_name] = platform
                if platform.side is not None:
                    self.side_info[platform.side].append(platform_name)
        elif 'platformName' in data:
            platform_name = data['platformName']
            if platform_name in self.platform_info:
                if isinstance(data, dict):
                    self.platform_info[platform_name].update(data.items())
                elif data is not None:
                    self.platform_info[platform_name].update(data)
        if 'track_list' in data:
            track_list = data['track_list']
            if track_list is not None:
                if len(track_list) > 0:
                    track_data = track_list[0]
                    if 'platform' in track_data:
                        track_platform = track_data['platform']
                        self_platform = data['name']
                        has_track_platform = track_platform in self.track_dict
                        has_self_platform = self_platform in self.track_dict
                        if has_track_platform:
                            self.track_dict[track_platform]['is_fired_list'].append(data['name'])
                        else:
                            track_platform_dict = {'is_fired_list': [data['name']], 'engage_target': None}
                            self.track_dict[track_platform] = track_platform_dict
                        if data['side'] != self.user_side:
                            track_platform = None
                        if has_self_platform:
                            self.track_dict[self_platform]['engage_target'] = track_platform
                        else:
                            missile_dict = {'is_fired_list': [], 'engage_target': track_platform}
                            self.track_dict[data['name']] = missile_dict
                    # print(self.track_dict)
        if 'id' in data:
            pid = data['id']
            if isinstance(pid, int):
                self.id_name_dict[pid] = data['name']

    def remove_platform(self, data):
        if 'name' in data:
            name = data['name']
            if name in self.platform_info:
                if self.platform_info[name].side is not None:
                    self.broken_platform_info[self.platform_info[name].side].append(name)
                self.platform_info.pop(name)
            if name in self.track_dict:
                self.track_dict.pop(name)
            for key, value in self.track_info.items():
                if name in value:
                    self.track_info[key].pop(name)
                    break
            self.remove_from_track_dict(name)

    def update_weapon_quantity(self, data):
        if 'launchPlatformName' in data:
            name = data['launchPlatformName']
            if name in self.platform_info:
                self.platform_info[name].weapons[0]['quantity'] -= 1

    def add_or_update_track(self, data):
        if 'platformName' in data:
            name = data['platformName']
            target_name = data['targetPlatformName']
            if name in self.platform_info and target_name in self.platform_info:
                self_platform = self.platform_info[name]
                track_platform = self.platform_info[target_name]
                if self_platform.side == 'red':
                    if track_platform.side == 'red':
                        if track_platform.type != '导弹':
                            return
                    if name in self.track_info['red']:
                        if target_name in self.track_info['red'][name]:
                            self.track_info['red'][name][target_name].update(data.items())
                        else:
                            self.track_info['red'][name][target_name] = Track(data.items(), track_platform)
                    else:
                        temp_dict = {target_name : Track(data.items(), track_platform)}
                        self.track_info['red'][name] = temp_dict
                else:
                    if track_platform.side == 'blue':
                        if track_platform.type != '导弹':
                            return
                    if name in self.track_info['blue']:
                        if target_name in self.track_info['blue'][name]:
                            self.track_info['blue'][name][target_name].update(data.items())
                        else:
                            self.track_info['blue'][name][target_name] = Track(data.items(), track_platform)
                    else:
                        temp_dict = {target_name : Track(data.items(), track_platform)}
                        self.track_info['blue'][name] = temp_dict

    def remove_track(self, data):
        if 'targetPlatformName' in data:
            name = data['platformName']
            target_name = data['targetPlatformName']
            self_platform = self.platform_info[name]
            if self_platform.side == 'red':
                red_team_track = self.track_info['red']
                if name in red_team_track:
                    if target_name in red_team_track[name]:
                        self.track_info['red'][name].pop(target_name)
            elif self_platform.side == 'blue':
                blue_team_track = self.track_info['blue']
                if name in blue_team_track:
                    if target_name in blue_team_track[name]:
                        self.track_info['blue'][name].pop(target_name)

    def remove_from_track_dict(self, name):
        for key, values in self.track_dict.items():
            if name in values['is_fired_list']:
                values['is_fired_list'].remove(name)
                return

    def to_dict(self):
        dic = {}
        if isinstance(self.header, Header):
            dic['header'] = self.header.to_dict()
        red_team = {'side': 'red', 'platform_list':[], 'track_list':[], 'broken_list':[]}
        blue_team = {'side': 'blue', 'platform_list':[], 'track_list':[], 'broken_list':[]}

        # 添加平台信息
        if len(self.platform_info) > 0:
            for value in self.platform_info.values():
                if value.type != '导弹':
                    if value.side == 'red':
                        red_team['platform_list'].append(value.to_dict())
                    else:
                        blue_team['platform_list'].append(value.to_dict())

        # 添加track信息
        for key in self.track_info:
            value = self.track_info[key]
            if len(value) > 0:
                if key == 'red':
                    if len(value) > 0:
                        track_id_list = []
                        for own_name, track_dic in value.items():
                            if len(track_dic) > 0:
                                for track_name, track in track_dic.items():
                                    if track_name in self.track_dict:
                                        track_data = self.track_dict[track_name]
                                        track.is_fired_num = len(track_data['is_fired_list'])
                                        track.engage_target = track_data['engage_target']
                                    else:
                                        track.is_fired_num = 0
                                        track.engage_target = None
                                    if track.target_id not in track_id_list:
                                        track_id_list.append(track.target_id)
                                        red_team['track_list'].append(track.to_dict())
                elif key == 'blue':
                    if len(value) > 0:
                        track_id_list = []
                        for own_name, track_dic in value.items():
                            if len(track_dic) > 0:
                                for track_name, track in track_dic.items():
                                    if track_name in self.track_dict:
                                        track_data = self.track_dict[track_name]
                                        track.is_fired_num = len(track_data['is_fired_list'])
                                        track.engage_target = track_data['engage_target']
                                    else:
                                        track.is_fired_num = 0
                                        track.engage_target = None
                                    if track.target_id not in track_id_list:
                                        track_id_list.append(track.target_id)
                                        blue_team['track_list'].append(track.to_dict())

        #添加战损平台信息
        for key in self.broken_platform_info:
            value = self.broken_platform_info[key]
            if len(value) > 0:
                if key == 'red':
                    red_team['broken_list'] = value
                elif key == 'blue':
                    blue_team['broken_list'] = value

        dic['side_list'] = [red_team, blue_team]

        #移除多余属性
        if 'broken_platform_info' in dic:
            dic.pop('broken_platform_info')
        if 'platform_info' in dic:
            dic.pop('platform_info')
        if 'track_info' in dic:
            dic.pop('track_info')
        if 'side_info' in dic:
            dic.pop('side_info')
        if 'id_name_dict' in dic:
            dic.pop('id_name_dict')
        return dic