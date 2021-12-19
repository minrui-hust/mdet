class Pointcloud(object):
    def __init__(self, points):
        super().__init__()

        r'''
        shape: N x F, float32
        encode: [x, y, z, other feature]
        '''
        self.points = points
