class YxRegistry:
    _functions = {}

    @classmethod
    def register(cls, name=None, func_type=None): #type 0:单个, 1:多个, 2:通用
        def decorator(func):
            key = name or func.__name__
            cls._functions[key] = {
                'func': func,
                'type': func_type
            }
            return func
        return decorator

    @classmethod
    def call(cls, name, targets, obs_data):
        if name not in cls._functions:
            raise KeyError('%s is not registered' % name)
        else:
            return cls._functions[name]['func'](targets, obs_data, True)

    @classmethod
    def call_kwargs(cls, name, targets, obs_data, **kwargs):
        if name not in cls._functions:
            raise KeyError('%s is not registered' % name)
        else:
            return cls._functions[name](targets, obs_data, **kwargs)

    @classmethod
    def get_functions(cls):
        task_list = {'single': [], 'multi': []}
        for key, value in cls._functions.items():
            func_type = value['type']
            if func_type == 0:
                task_list['single'].append(key)
            elif func_type == 1:
                task_list['multi'].append(key)
            elif func_type == 2:
                task_list['single'].append(key)
                task_list['multi'].append(key)
        return task_list