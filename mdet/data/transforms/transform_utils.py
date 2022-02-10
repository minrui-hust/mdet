import numba
import numpy as np

from mdet.core.geometry2d import boxes_corners, rotate_points_jit, box_collision_test_jit, rotate_points


def noise_per_box(boxes, loc_noises, rot_noises):
    boxes = boxes[:, [0, 1, 3, 4, 6, 7]]
    box_centers = boxes[:, :2]
    box_corners = boxes_corners(boxes)  # Nx4x2
    noise_index = noise_per_box_jit(
        box_centers, box_corners, loc_noises, rot_noises)
    return noise_index


@numba.njit
def noise_per_box_jit(box_centers, box_corners, loc_noises, rot_noises):
    num_boxes = box_centers.shape[0]
    num_tests = loc_noises.shape[1] - 1
    current_corners = np.zeros((4, 2), dtype=box_corners.dtype)
    noise_index = np.full(num_boxes, num_tests, dtype=np.int64)

    for i in range(num_boxes):
        for j in range(num_tests):
            current_corners = rotate_points_jit(
                box_corners[i] - box_centers[i], rot_noises[i, j]) + box_centers[i] + loc_noises[i, j, :2]
            coll_mat = box_collision_test_jit(
                current_corners.reshape(1, 4, 2), box_corners)
            coll_mat[0, i] = False  # mask self
            if not coll_mat.any():  # pass collision test
                noise_index[i] = j  # record the noise index added
                box_corners[i] = current_corners  # update corner for remains
                break  # break once succeed, else default value will be used, which means no noise added
    return noise_index


def transform_points(points, centers, point_indice_list, translation, rotation):
    """Apply transforms to points and box centers.

    Args:
        points (np.ndarray): Input points.
        centers (np.ndarray): Input box centers.
        point_masks (np.ndarray): Mask to indicate which points need
            to be transformed.
        translation(np.ndarray): Location transform to be applied.
        rotation(np.ndarray): Rotation transform to be applied.
    """
    num_box = centers.shape[0]

    for i in range(num_box):
        box_points = points[point_indice_list[i], :3]
        box_points = box_points - centers[i]
        box_points[:, :2] = rotate_points(box_points[:, :2], rotation[i])
        box_points = box_points + centers[i] + translation[i]
        points[point_indice_list[i], :3] = box_points


def transform_boxes(boxes, translation, rotation):
    """Transform 3D boxes.

    Args:
        boxes (np.ndarray): 3D boxes to be transformed.
        translation(np.ndarray): Location transform to be applied.
        rotation(np.ndarray): Rotation transform to be applied.
    """
    boxes[:, :3] = boxes[:, :3] + translation
    boxes[:, 6:] = rotate_points(boxes[:, 6:], rotation)
