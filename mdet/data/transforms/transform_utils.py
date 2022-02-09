import numba
import numpy as np

from mdet.core.box_np_ops import box_bev_corner, _box_collision_test, _rotation_box2d_jit


def noise_per_box(boxes, loc_noises, rot_noises):
    box_centers = boxes[:, :2]
    box_corners = box_bev_corner(boxes)  # Nx4x2
    noise_index = noise_per_box_jit(
        box_centers, box_corners, loc_noises, rot_noises)
    return noise_index


@numba.njit
def noise_per_box_jit(box_centers, box_corners, loc_noises, rot_noises):
    num_boxes = box_centers.shape[0]
    num_tests = loc_noises.shape[1] - 1
    current_corners = np.zeros((4, 2), dtype=box_corners.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=box_corners.dtype)
    noise_index = np.full(num_boxes, num_tests, dtype=np.int64)

    for i in range(num_boxes):
        for j in range(num_tests):
            current_corners[:] = box_corners[i] - box_centers[i]
            _rotation_box2d_jit(current_corners, rot_noises[i, j], rot_mat_T)
            current_corners += box_centers[i] + loc_noises[i, j, :2]
            coll_mat = _box_collision_test(
                current_corners.reshape(1, 4, 2), box_corners)
            coll_mat[0, i] = False  # mask self
            if not coll_mat.any():  # pass collision test
                noise_index[i] = j  # record the noise index added
                box_corners[i] = current_corners  # update corner for remains
                break  # break once succeed, else default value will be used, which means no noise added
    return noise_index


@numba.njit
def transform_points(points, centers, point_masks, translation, rotation):
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
    num_points = points.shape[0]

    for i in range(num_points):
        for j in range(num_box):
            if point_masks[i, j]:
                points[i, :3] -= centers[j]
                points[i, 0] = points[i, 0] * \
                    rotation[j, 0] - points[i, 1]*rotation[j, 1]
                points[i, 1] = points[i, 0] * \
                    rotation[j, 1] + points[i, 1]*rotation[j, 0]
                points[i, :3] = points[i, :3] + centers[j] + translation[j]
                break  # only apply first box's transform


@numba.njit
def transform_boxes(boxes, translation, rotation):
    """Transform 3D boxes.

    Args:
        boxes (np.ndarray): 3D boxes to be transformed.
        translation(np.ndarray): Location transform to be applied.
        rotation(np.ndarray): Rotation transform to be applied.
    """
    num_box = boxes.shape[0]

    for i in range(num_box):
        boxes[i, :3] += translation[i]
        boxes[i, 6] = boxes[i, 6]*rotation[i, 0] - boxes[i, 7] * rotation[i, 1]
        boxes[i, 7] = boxes[i, 6]*rotation[i, 1] + boxes[i, 7] * rotation[i, 0]
