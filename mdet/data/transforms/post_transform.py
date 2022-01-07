from mdet.utils.factory import FI


@FI.register
class WhiteList(object):
    r'''
    element not int WhiteList will be deleted
    '''

    def __init__(self, white_list=[]):
        super().__init__()
        self.white_list = white_list

    def __call__(self, sample, info):
        return sample.select(self.white_list)
