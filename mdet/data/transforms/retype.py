from mdet.utils.factory import FI
import numpy as np
from mdet.core.annotation import Annotation3d


@FI.register
class WaymoCarRetyper(object):
    r'''
    Retyper is used in two aspects:
    1. dataset transform
    2. database transform
    '''

    def __init__(self, raw_car_type, new_car_types, other_types):
        self.raw_car_type = raw_car_type
        self.new_car_types = new_car_types
        self.other_types = other_types  # rawtype: newtype

    def retype_anno(self, anno):
        new_types = [self.new_type(type, box)
                     for (type, box) in zip(anno.types, anno.boxes)]
        new_types = np.array(new_types, dtype=np.int32)
        return Annotation3d(boxes=anno.boxes, types=new_types, scores=anno.scores, num_points=anno.num_points)

    def retype_db_info(self, db_info):
        new_db_info = {}
        for type, info_list in db_info.items():
            if type in self.other_types:
                new_type_id = self.other_types[type]
                new_db_info[new_type_id] = info_list
            elif type == self.raw_car_type:
                for info in info_list:
                    new_type_id = self.new_type(type, info['box'])
                    if new_type_id not in new_db_info:
                        new_db_info[new_type_id] = [info]
                    else:
                        new_db_info[new_type_id].append(info)
            else:
                pass  # just ignore types we not care
        return new_db_info

    def get_type_map(self):
        r'''
        map the new type id back to original type id, used in dataset.format for evaluation
        '''
        map_new_to_raw = {}
        for raw_type, new_type in self.other_types.items():
            map_new_to_raw[new_type] = raw_type
        for new_type in self.new_car_types.keys():
            map_new_to_raw[new_type] = self.raw_car_type

        return map_new_to_raw

    def new_type(self, raw_type, box):
        if raw_type in self.other_types:
            return self.other_types[raw_type]
        elif raw_type == self.raw_car_type:
            length = box[3] * 2
            for new_type_id, (min_length, max_length) in self.new_car_types.items():
                if length >= min_length and length < max_length:
                    return new_type_id
            assert False, 'Unknow vehicle size'
        assert False, 'Unknow raw_type'
