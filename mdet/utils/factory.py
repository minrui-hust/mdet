import inspect


def Singleton(cls):
    _instance = {}

    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]
    return inner


@Singleton
class Factory(object):
    def __init__(self):
        self._component_dict = {}
        pass

    def register(self):

        def _register(cls):
            self._register(cls)
            return cls

        return _register

    def create(self, cfg):
        if not isinstance(cfg, dict):
            raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
        if 'type' not in cfg:
            raise KeyError(f'`cfg` contain the key "type", but got {cfg}')

        args = cfg.copy()
        component_name = args.pop('type')
        if component_name not in self._component_dict:
            raise KeyError(f'{component_name} is not registered')

        return self._component_dict[component_name](**args)

    def _register(self, component_cls):
        if not inspect.isclass(component_cls):
            raise TypeError(
                f'component must be a class, but got {type(component_cls)}')

        component_name = component_cls.__name__
        if component_name in self._component_dict:
            raise KeyError(f'{component_name} is already registered')

        self._component_dict[component_name] = component_cls


FI = Factory()


if __name__ == '__main__':

    @FI.register()
    class A(object):
        def __init__(self, c0, c1=5):
            self.c0 = c0
            self.c1 = c1

        def say(self):
            print(self.c0, self.c1)

    a = FI.create({'type': 'A', 'c0': 2, 'c1': 8})

    a.say()

    cf = Factory()

    @FI.register()
    class B(object):
        def __init__(self, c0, c1=5):
            self.c0 = c0
            self.c1 = c1

        def say(self):
            print(self.c0, self.c1)

    b = FI.create({'type': 'B', 'c0': 3, 'c1': 9})
    b.say()
