from mdet.core.box_np_ops import points_in_box, rotate3d_z
import mdet.utils.io as io
from mdet.data.datasets.waymo_det3d.summary import normlize_boxes
import numpy as np
from mdet.utils.viz import Visualizer


object_name = '3UJet22LcSM39E7W52UwZA'
pcd_path = '/data/tmp/waymo/training/pcds/10017090168044687777_6380_000_6400_000-2.pkl'
anno_path = '/data/tmp/waymo/training/annos/10017090168044687777_6380_000_6400_000-2.pkl'

pcd = io.load(pcd_path, compress=True)
anno = io.load(anno_path)
objects = anno['objects']
objects_box_list = [object['box'] for object in objects if object['name']=='3UJet22LcSM39E7W52UwZA']
objects_box = normlize_boxes(np.stack(objects_box_list, axis=0))

indices = points_in_box(pcd, objects_box)
pcd=pcd[indices[:,0]]
indices = points_in_box(pcd, objects_box)

local_pcd = rotate3d_z(pcd[np.newaxis,:,:3] - objects_box[0][:3], np.stack([objects_box[:,6], objects_box[:, 7]], axis=-1))

mask = (local_pcd[0,:,0] > objects_box[0][3]) | (local_pcd[0,:,0] < -objects_box[0][3]) | (local_pcd[0,:,1] > objects_box[0][4]) | (local_pcd[0,:,1] < -objects_box[0][4]) | (local_pcd[0,:,2] > objects_box[0][5]) | (local_pcd[0,:,2] < -objects_box[0][5])
invalid_index = np.where(mask)

print(local_pcd.shape)
print(invalid_index[0].shape)
print(invalid_index)

vis = Visualizer()
points = np.concatenate([pcd[indices[:, i]] for i in range(len(objects_box))])
vis.add_points(points)
vis.add_box(objects_box, box_label=[object['name'] for object in objects if object['name']=='3UJet22LcSM39E7W52UwZA'])
vis.show()
