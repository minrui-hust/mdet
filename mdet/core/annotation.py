
class Annotation3d(object):
    def __init__(self, boxes, types, scores=None, num_points=None):
        super().__init__()

        r'''
        shape: N x 7, float32
        encode: [center_x, center_y, center_z, size_x, size_y, size_z, rot]
        rot: the box x axis relative to global x axis in counter-clock wise direction
        '''
        self.boxes = boxes

        r'''
        shape: N, int32
        type id in [0, type_num)
        '''
        self.types = types

        r'''
        optional
        shape: N, float32
        indicate box confidence
        '''
        self.scores = scores

        r'''
        optional
        shape: N, int32
        indicate how many points in each box
        '''
        self.num_points = num_points

    def __getitem__(self, idx):
        return Annotation3d(
            boxes=self.boxes[idx],
            types=self.types[idx],
            scores=None if self.scores is None else self.scores[idx],
            num_points=None if self.num_points is None else self.num_points[idx],
        )

    def __repr__(self):
        s = 'Annotation3d{\n'
        s += f'boxes: {self.boxes},\ntypes:\n{self.types}'
        if self.scores is not None:
            s += f',\nscores:\n{self.scores}'
        if self.num_points is not None:
            s += f',\nnum_points:\n{self.num_points}'
        s += '\n}'
        return s
