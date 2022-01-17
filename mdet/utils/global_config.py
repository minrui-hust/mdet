from mdet.utils.singleton import Singleton


@Singleton
class GlobalConfig(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, key):
        if key not in self:
            return None
        else:
            return dict.__getitem__(self, key)


GCFG = GlobalConfig()
