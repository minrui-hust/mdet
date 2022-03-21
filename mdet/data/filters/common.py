from mai.utils import FI


@FI.register
class IntervalDownsampler(object):
    def __init__(self, interval=1):
        super().__init__()
        self.interval = interval

    def __call__(self, info_list):
        return info_list[::self.interval]

