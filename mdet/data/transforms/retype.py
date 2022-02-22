from mdet.utils.factory import FI

@FI.register
class WaymoCarRetyper(object):
    def __init__(self, vehicle_types=[(0.1, 5.0, 9)]): #(min_len, max_len, type_id)
        pass

    def retype_anno(self, anno):
        pass

    def retype_db_info(self, db_info):
        pass

    def get_type_map(self):
        pass
