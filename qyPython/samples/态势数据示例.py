observation = \
{
     "header": {
          "sim_time": 249.4,                                     # 仿真时间（单位：秒）
          "real_time": 1764492528.347569,                        # 系统时间（单位：秒）
          "is_active": True,                                     # 是否在推演中
          "time_ratio": 1                                        # 时间倍率
     },                         # 仿真推演头信息
     "side_list": [
          {
               "side": "red",                 # 红方阵营
               "platform_list": [
                    {
                         "id": 1,                                # 平台id
                         "side": "red",                          # 阵营方
                         "type": "无人机",                        # 平台类型
                         "speed": 447.0441,                      # 速度（单位：米/秒）
                         "name": "红无人机1",                      # 平台名称
                         "longitude": 145.9127479,               # 飞行经度（单位：度）
                         "latitude": 33.4996353,                 # 飞行纬度（单位：度）
                         "altitude": 3459.61,                    # 飞行海拔（单位：米）
                         "heading": -2.970339202054858,          # 偏航角（单位：弧度）. 取值范围(-π,π). 0是正北，顺时针为正，逆时针为负
                         "pitch": 0.0010105456369047167,         # 俯仰角（单位：弧度）. 取值范围(-π/2,π/2). 0是平视，-π/2为下，π/2为上
                         "roll": -1.4009496106370685,            # 横滚角（单位：弧度）. 取值范围(-π,π). 0是水平，负值为左滚，正值为右滚
                         "velocity_x": -440.50448810171736,      # 速度x分量（单位：米/秒）. 东为正，西为负
                         "velocity_y": -76.18405462949177,       # 速度y分量（单位：米/秒）. 北为正，南为负
                         "velocity_z": -0.4518698739544384,      # 速度z分量（单位：米/秒）. 上为正，下为负
                         "is_controlled": None,
                         "mover": {
                              "id": 0,
                              "name": ""
                         },
                         "sensors": [
                              {
                                   "id": 4,
                                   "name": "雷达",
                                   "type": "WSF_RADAR_SENSOR",
                                   "turned_on": True,       # 雷达是否开启
                                   "az_max": 60,            # 水平扫描角度范围（单位：度）
                                   "az_min": -120,          # 水平扫描角度范围（单位：度）
                                   "el_max": 40,            # 垂直扫描角度范围（单位：度）
                                   "el_min": -40,           # 垂直扫描角度范围（单位：度）
                                   "max_range": 100000      # 最大检测范围（单位：米）
                              }
                         ],
                         "weapons": [
                              {
                                   "id": 5,
                                   "name": "导弹",
                                   "launched_platform_type": "导弹",
                                   "type": "WSF_EXPLICIT_WEAPON",
                                   "quantity": 2                 # 平台挂载武器剩余数量
                              }
                         ]
                    },
                    {
                         "id": 25,
                         "side": "red",
                         "type": "无人机",
                         "speed": 447.0441,
                         "name": "红无人机4",
                         "longitude": 145.9366916,
                         "latitude": 33.3190407,
                         "altitude": 3491.57,
                         "heading": 1.9889841968897461,
                         "pitch": 0.0003333578871309169,
                         "roll": -1.4009496106370685,
                         "velocity_x": -181.5471908748407,
                         "velocity_y": 408.52051810461455,
                         "velocity_z": -0.1489191683391236,
                         "is_controlled": None,
                         "mover": {
                              "id": 0,
                              "name": ""
                         },
                         "sensors": [
                              {
                                   "id": 28,
                                   "name": "雷达",
                                   "type": "WSF_RADAR_SENSOR",
                                   "turned_on": False,
                                   "az_max": 60,
                                   "az_min": -60,
                                   "el_max": 40,
                                   "el_min": -40,
                                   "max_range": 15000
                              }
                         ],
                         "weapons": [
                              {
                                   "id": 29,
                                   "name": "导弹",
                                   "launched_platform_type": "导弹",
                                   "type": "WSF_EXPLICIT_WEAPON",
                                   "quantity": 2
                              }
                         ]
                    },
                    {
                         "id": 73,
                         "side": "red",
                         "type": "有人机",
                         "speed": 447.0441,
                         "name": "红有人机2",
                         "longitude": 145.8943471,
                         "latitude": 33.1283451,
                         "altitude": 3551.63,
                         "heading": 2.3241973690155326,
                         "pitch": 0.030387927606473273,
                         "roll": -1.4009496106370685,
                         "velocity_x": -305.69222747102214,
                         "velocity_y": 325.90827225970656,
                         "velocity_z": -13.582287746376235,
                         "is_controlled": None,
                         "mover": {
                              "id": 0,
                              "name": ""
                         },
                         "sensors": [
                              {
                                   "id": 76,
                                   "name": "雷达",
                                   "type": "WSF_RADAR_SENSOR",
                                   "turned_on": False,
                                   "az_max": 180,
                                   "az_min": -180,
                                   "el_max": 60,
                                   "el_min": -60,
                                   "max_range": 25000
                              }
                         ],
                         "weapons": [
                              {
                                   "id": 77,
                                   "name": "导弹",
                                   "launched_platform_type": "导弹",
                                   "type": "WSF_EXPLICIT_WEAPON",
                                   "quantity": 4
                              }
                         ]
                    }
               ],        # 红方现存平台信息
"track_list": [
     {
          "target_id": 281,                       # 探测到单位id（我方发射的导弹、敌方平台、敌方发射的导弹）
          "target_name": "红有人机1_导弹_4",         # 单位的名称
          "platform_entity_type": "导弹",          # 单位的类型（导弹/无人机/有人机）
          "platform_entity_side": "red",          # 单位的所属方（red/blue）
          "longitude": 146.0974,                  # 单位的飞行经度（单位：度）
          "latitude": 33.2581,                    # 单位的飞行纬度（单位：度）
          "altitude": 3891.9141,                  # 单位的飞行高度（单位：米）
          "speed": 1200,                          # 单位的飞行速度（单位：米/秒）
          "v_x": -1129.1733935524214,             # 单位的飞行速度x分量（单位：米/秒）. 东为正，西为负
          "v_y": 406.0398080310301,               # 单位的飞行速度y分量（单位：米/秒）. 北为正，南为负
          "v_z": -9.95598249458385,               # 单位的飞行速度z分量（单位：米/秒）. 上为正，下为负
          "heading": 2.796399688801103,           # 单位的偏航角（单位：弧度）. 取值范围(-π,π). 0是正北，顺时针为正，逆时针为负
          "pitch": 0.008297295263981043,          # 单位的俯仰角（单位：弧度）. 取值范围(-π/2,π/2). 0是平视，-π/2为下，π/2为上
          "roll": 0,                              # 单位的横滚角（单位：弧度）. 取值范围(-π,π). 0是水平，负值为左滚，正值为右滚
          "is_fired_num": 0,                      # 该单位当前被我方导弹攻击的数量
          "engage_target": "蓝有人机2"              # 该单位交战的敌方目标（仅我方导弹有该属性）
     },
     {
          "target_id": 291,
          "target_name": "蓝有人机2_导弹_1",
          "platform_entity_type": "导弹",
          "platform_entity_side": "blue",
          "longitude": 146.0727,
          "latitude": 33.2519,
          "altitude": 3649.2267,
          "speed": 1200,
          "v_x": -487.03675896266066,
          "v_y": -1096.582442654307,
          "v_z": -17.3822288979417,
          "heading": -1.9887660307332466,
          "pitch": 0.014486232791552934,
          "roll": 0,
          "is_fired_num": 0,
          "engage_target": None
     },
     {
          "target_id": 89,
          "target_name": "蓝无人机2",
          "platform_entity_type": "无人机",
          "platform_entity_side": "blue",
          "longitude": 146.2412,
          "latitude": 33.3431,
          "altitude": 3500.4212,
          "speed": 300,
          "v_x": 286.3840586222374,
          "v_y": 89.35418755052243,
          "v_z": -0.011586239669618885,
          "heading": 0.3024376341025854,
          "pitch": 0.00003839724354387525,
          "roll": 1.4009496106370685,
          "is_fired_num": 1,
          "engage_target": None
     }
],           # 红方当前探测的单位信息
               "broken_list": [
                    "红无人机2_导弹_2",
                    "红无人机1_导弹_1",
                    "红无人机2",
               ]           # 红方已损失的单位信息
          },
          {
               "side": "blue",                # 蓝方阵营
               "platform_list": [
                    {
                         "id": 81,
                         "side": "blue",
                         "type": "无人机",
                         "speed": 417.6241,
                         "name": "蓝无人机1",
                         "longitude": 146.2653829,
                         "latitude": 33.2359969,
                         "altitude": 3421.23,
                         "heading": -1.6574536694856672,
                         "pitch": 0.005220279792715039,
                         "roll": 1.4009496106370685,
                         "velocity_x": -36.14414856810012,
                         "velocity_y": -416.0514048163323,
                         "velocity_z": -2.1797672771957433,
                         "is_controlled": None,
                         "mover": {
                              "id": 0,
                              "name": ""
                         },
                         "sensors": [
                              {
                                   "id": 84,
                                   "name": "雷达",
                                   "type": "WSF_RADAR_SENSOR",
                                   "turned_on": False,
                                   "az_max": 60,
                                   "az_min": -60,
                                   "el_max": 40,
                                   "el_min": -40,
                                   "max_range": 15000
                              }
                         ],
                         "weapons": [
                              {
                                   "id": 85,
                                   "name": "导弹",
                                   "launched_platform_type": "导弹",
                                   "type": "WSF_EXPLICIT_WEAPON",
                                   "quantity": 2
                              }
                         ]
                    },
                    {
                         "id": 89,
                         "side": "blue",
                         "type": "无人机",
                         "speed": 417.6241,
                         "name": "蓝无人机2",
                         "longitude": 146.1724288,
                         "latitude": 33.5075105,
                         "altitude": 3405.73,
                         "heading": -1.3174338605998879,
                         "pitch": 0.008527678725244294,
                         "roll": 1.4009496106370685,
                         "velocity_x": 104.67780382624824,
                         "velocity_y": -404.276883605786,
                         "velocity_z": -3.5610502448427273,
                         "is_controlled": None,
                         "mover": {
                              "id": 0,
                              "name": ""
                         },
                         "sensors": [
                              {
                                   "id": 92,
                                   "name": "雷达",
                                   "type": "WSF_RADAR_SENSOR",
                                   "turned_on": False,
                                   "az_max": 60,
                                   "az_min": -60,
                                   "el_max": 40,
                                   "el_min": -40,
                                   "max_range": 15000
                              }
                         ],
                         "weapons": [
                              {
                                   "id": 93,
                                   "name": "导弹",
                                   "launched_platform_type": "导弹",
                                   "type": "WSF_EXPLICIT_WEAPON",
                                   "quantity": 2
                              }
                         ]
                    },
                    {
                         "id": 121,
                         "side": "blue",
                         "type": "无人机",
                         "speed": 417.6241,
                         "name": "蓝无人机6",
                         "longitude": 146.1464338,
                         "latitude": 33.1357368,
                         "altitude": 3424.85,
                         "heading": -0.30269419750262855,
                         "pitch": 0.001736602605734358,
                         "roll": -1.4009496106370685,
                         "velocity_x": 398.63698364831276,
                         "velocity_y": -124.49077071336227,
                         "velocity_z": -0.724972900926424,
                         "is_controlled": None,
                         "mover": {
                              "id": 0,
                              "name": ""
                         },
                         "sensors": [
                              {
                                   "id": 124,
                                   "name": "雷达",
                                   "type": "WSF_RADAR_SENSOR",
                                   "turned_on": False,
                                   "az_max": 60,
                                   "az_min": -60,
                                   "el_max": 40,
                                   "el_min": -40,
                                   "max_range": 15000
                              }
                         ],
                         "weapons": [
                              {
                                   "id": 125,
                                   "name": "导弹",
                                   "launched_platform_type": "导弹",
                                   "type": "WSF_EXPLICIT_WEAPON",
                                   "quantity": 2
                              }
                         ]
                    },
                    {
                         "id": 137,
                         "side": "blue",
                         "type": "无人机",
                         "speed": 394.1438,
                         "name": "蓝无人机8",
                         "longitude": 146.199069,
                         "latitude": 33.1172653,
                         "altitude": 3440.15,
                         "heading": -1.969586607583081,
                         "pitch": 0.004024729255098924,
                         "roll": 0,
                         "velocity_x": -153.04627771969544,
                         "velocity_y": -363.21300650811673,
                         "velocity_z": -1.5859979274410803,
                         "is_controlled": None,
                         "mover": {
                              "id": 0,
                              "name": ""
                         },
                         "sensors": [
                              {
                                   "id": 140,
                                   "name": "雷达",
                                   "type": "WSF_RADAR_SENSOR",
                                   "turned_on": False,
                                   "az_max": 60,
                                   "az_min": -60,
                                   "el_max": 40,
                                   "el_min": -40,
                                   "max_range": 15000
                              }
                         ],
                         "weapons": [
                              {
                                   "id": 141,
                                   "name": "导弹",
                                   "launched_platform_type": "导弹",
                                   "type": "WSF_EXPLICIT_WEAPON",
                                   "quantity": 2
                              }
                         ]
                    },
                    {
                         "id": 145,
                         "side": "blue",
                         "type": "有人机",
                         "speed": 417.6241,
                         "name": "蓝有人机1",
                         "longitude": 146.1966968,
                         "latitude": 33.3761638,
                         "altitude": 3598.06,
                         "heading": -2.3215200339429733,
                         "pitch": 0.04898441078647285,
                         "roll": -1.4009496106370685,
                         "velocity_x": -284.5481470298771,
                         "velocity_y": -304.99856329288036,
                         "velocity_z": -20.448733992145304,
                         "is_controlled": None,
                         "mover": {
                              "id": 0,
                              "name": ""
                         },
                         "sensors": [
                              {
                                   "id": 148,
                                   "name": "雷达",
                                   "type": "WSF_RADAR_SENSOR",
                                   "turned_on": False,
                                   "az_max": 180,
                                   "az_min": -180,
                                   "el_max": 60,
                                   "el_min": -60,
                                   "max_range": 25000
                              }
                         ],
                         "weapons": [
                              {
                                   "id": 149,
                                   "name": "导弹",
                                   "launched_platform_type": "导弹",
                                   "type": "WSF_EXPLICIT_WEAPON",
                                   "quantity": 4
                              }
                         ]
                    }
               ],        # 蓝方现存平台信息
               "track_list": [
                    {
                         "target_id": 256,
                         "target_name": "蓝无人机6_导弹_2",
                         "platform_entity_type": "导弹",
                         "platform_entity_side": "blue",
                         "longitude": 146.0975,
                         "latitude": 33.1529,
                         "altitude": 2351.5743,
                         "speed": 1200,
                         "v_x": 936.6385575563561,
                         "v_y": -738.0399556033535,
                         "v_z": 134.18359225981698,
                         "heading": -0.6673598007558198,
                         "pitch": -0.11205362863653995,
                         "roll": 0,
                         "is_fired_num": 0,
                         "engage_target": None
                    }
               ],           # 蓝方当前探测的单位信息
               "broken_list": [
                    "蓝无人机3",
                    "蓝无人机3_导弹_1",
                    "蓝有人机2",
                    "蓝有人机2_导弹_3",
                    "蓝有人机2_导弹_4",
                    "蓝无人机3_导弹_2"
               ]           # 蓝方已损失的单位信息
          }
     ]                       # 双方阵营列表
}


