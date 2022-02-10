import numba as nb
import numpy as np

from mdet.core.geometry2d import rotate_points
from scipy.spatial import KDTree


def points_in_boxes(points, boxes):
    r'''
    check if point in bev box(3d box with only yaw)
    Args:
        points: [N,3]
        boxes: [M, 8]
    '''
    num_points = points.shape[0]
    num_boxes = boxes.shape[0]
    ret = np.zeros((num_points, num_boxes), dtype=np.bool_)

    kdt = KDTree(np.ascontiguousarray(
        points[:, :2]), leafsize=100, balanced_tree=False, compact_nodes=False)

    center = boxes[:, :2]
    radius = np.linalg.norm(boxes[:, 3:5], axis=-1)
    nbs_list = kdt.query_ball_point(center, radius, return_sorted=False)

    indice_list = []
    for i in range(num_boxes):
        nbs = np.array(nbs_list[i], dtype=np.int32)
        nb_points = points[nbs, :3]

        box = boxes[i]
        pos = box[:3]
        extend = box[3:6]
        rot = np.array([box[6], -box[7]], dtype=box.dtype)

        local_points = np.empty((len(nbs), 3), dtype=points.dtype)
        local_points[:, :2] = rotate_points(nb_points[:, :2] - pos[:2], rot)
        local_points[:, 2] = nb_points[:, 2] - pos[2]

        local_points_abs = np.abs(local_points)

        in_mask = (local_points_abs[:, 0] < extend[0]) & (
            local_points_abs[:, 1] < extend[1]) & (local_points_abs[:, 2] < extend[2])

        in_indice = nbs[in_mask]

        indice_list.append(in_indice)

    return indice_list
