import numba as nb
import numpy as np

r'''
geometry2d operate on 2d points, lines, boxes
points: of shape [2], [N, 2] or [B, N, 2]
lines: of shape [4], [N, 4] or [B, N, 4]
boxes: of shape [6], [N, 6] or [B, N, 6], 
       single box is in fromat [center_x, center_y, extend_x, extend_y, cos_alpha, sin_alpha]
rotations: of shape [2], [M, 2]
'''


def rotate_points(points, rotation):
    r'''
    rotate points by rotation
    Args:
        points: [2], [N, 2] or [B, N, 2]
        rotation: [2], [M, 2]
    Return:
        rotated points
    '''

    points_ndim = len(points.shape)
    rotation_ndim = len(rotation.shape)

    if rotation_ndim == 1 and (points_ndim == 1 or points_ndim == 2 or points_ndim == 3):
        t = np.array([-rotation[1], rotation[0]], dtype=rotation.dtype)
        rotm = np.stack((rotation, t), axis=0)
        return points @ rotm
    elif rotation_ndim == 2 and (points_ndim == 2 or points_ndim == 3):
        t = np.stack((-rotation[:, 1], rotation[:, 0]), axis=-1)
        rotm = np.stack((rotation, t), axis=1)
        if points_ndim == 2:
            return (points[:, np.newaxis, :] @ rotm).squeeze(1)
        else:  # points_ndim==3
            return points @ rotm
    else:
        raise AssertionError('ndim combination invalid')


@nb.njit(fastmath=True)
def rotate_points_jit(points, rotation):
    r'''
    only handle points [N, 2], rotation [2]
    '''
    points_ndim = len(points.shape)
    rotation_ndim = len(rotation.shape)
    assert(points_ndim == 2 and rotation_ndim == 1)

    t = np.array([-rotation[1], rotation[0]], dtype=rotation.dtype)
    rotm = np.stack((rotation, t), axis=0)
    return np.ascontiguousarray(points) @ rotm


def boxes_corners(boxes):
    center, extend, rotation = boxes[:, :2], boxes[:, 2:4], boxes[:, 4:6]

    # get corner in body frame, shape Nx4x2
    corners_norm = np.stack(np.unravel_index(
        np.arange(4), [2] * 2), axis=1).astype(extend.dtype)
    corners_norm = corners_norm[[0, 1, 3, 2]]
    corners_norm = 2 * corners_norm - 1
    corners = extend.reshape(-1, 1, 2) * corners_norm.reshape(1, 4, 2)

    # transform corners into global frame
    corners = rotate_points(corners, rotation) + center[:, np.newaxis, :]

    return corners


def box_collision_test(kboxes, qboxes):
    kboxes_corners = boxes_corners(kboxes)
    qboxes_corners = boxes_corners(qboxes)
    return box_collision_test_jit(kboxes_corners, qboxes_corners)


@nb.njit(fastmath=True)
def box_collision_test_jit(boxes, qboxes, clockwise=True):
    """Box collision test.

    Args:
        boxes (np.ndarray): current standard boxes.
        qboxes (np.ndarray): standard Boxes to be avoid colliding.
        clockwise (bool): Whether the corners are in clockwise order.
            Default: True.
    """
    N = boxes.shape[0]
    K = qboxes.shape[0]
    ret = np.zeros((N, K), dtype=np.bool_)
    slices = np.array([1, 2, 3, 0])

    lines_boxes = np.stack((boxes, boxes[:, slices, :]),
                           axis=2)  # [N, 4, 2(line), 2(xy)]
    lines_qboxes = np.stack((qboxes, qboxes[:, slices, :]), axis=2)
    boxes_standup = corners_to_standup_jit(boxes)
    qboxes_standup = corners_to_standup_jit(qboxes)
    for i in range(N):
        for j in range(K):
            # calculate standup first
            iw = (min(boxes_standup[i, 2], qboxes_standup[j, 2]) -
                  max(boxes_standup[i, 0], qboxes_standup[j, 0]))
            if iw > 0:
                ih = (min(boxes_standup[i, 3], qboxes_standup[j, 3]) -
                      max(boxes_standup[i, 1], qboxes_standup[j, 1]))
                if ih > 0:
                    for k in range(4):
                        for box_l in range(4):
                            A = lines_boxes[i, k, 0]
                            B = lines_boxes[i, k, 1]
                            C = lines_qboxes[j, box_l, 0]
                            D = lines_qboxes[j, box_l, 1]
                            acd = (D[1] - A[1]) * (C[0] - A[0]) > (
                                C[1] - A[1]) * (D[0] - A[0])
                            bcd = (D[1] - B[1]) * (C[0] - B[0]) > (
                                C[1] - B[1]) * (D[0] - B[0])
                            if acd != bcd:
                                abc = (C[1] - A[1]) * (B[0] - A[0]) > (
                                    B[1] - A[1]) * (C[0] - A[0])
                                abd = (D[1] - A[1]) * (B[0] - A[0]) > (
                                    B[1] - A[1]) * (D[0] - A[0])
                                if abc != abd:
                                    ret[i, j] = True  # collision.
                                    break
                        if ret[i, j] is True:
                            break
                    if ret[i, j] is False:
                        # now check complete overlap.
                        # box overlap qbox:
                        box_overlap_qbox = True
                        for box_l in range(4):  # point l in qboxes
                            for k in range(4):  # corner k in boxes
                                vec = boxes[i, k] - boxes[i, (k + 1) % 4]
                                if clockwise:
                                    vec = -vec
                                cross = vec[1] * (boxes[i, k, 0] -
                                                  qboxes[j, box_l, 0])
                                cross -= vec[0] * (boxes[i, k, 1] -
                                                   qboxes[j, box_l, 1])
                                if cross >= 0:
                                    box_overlap_qbox = False
                                    break
                            if box_overlap_qbox is False:
                                break

                        if box_overlap_qbox is False:
                            qbox_overlap_box = True
                            for box_l in range(4):  # point box_l in boxes
                                for k in range(4):  # corner k in qboxes
                                    vec = qboxes[j, k] - qboxes[j, (k + 1) % 4]
                                    if clockwise:
                                        vec = -vec
                                    cross = vec[1] * (qboxes[j, k, 0] -
                                                      boxes[i, box_l, 0])
                                    cross -= vec[0] * (qboxes[j, k, 1] -
                                                       boxes[i, box_l, 1])
                                    if cross >= 0:  #
                                        qbox_overlap_box = False
                                        break
                                if qbox_overlap_box is False:
                                    break
                            if qbox_overlap_box:
                                ret[i, j] = True  # collision.
                        else:
                            ret[i, j] = True  # collision.
    return ret


@nb.njit(fastmath=True)
def corners_to_standup_jit(boxes_corner):
    """Convert boxes_corner to aligned (min-max) boxes.

    Args:
        boxes_corner (np.ndarray, shape=[N, 4, 2]): Boxes corners.

    Returns:
        np.ndarray, shape=[N, 4]: Aligned (min-max) boxes.
    """
    num_boxes = boxes_corner.shape[0]
    result = np.zeros((num_boxes, 4), dtype=boxes_corner.dtype)
    for i in range(num_boxes):
        for j in range(2):
            result[i, j] = np.min(boxes_corner[i, :, j])
        for j in range(2):
            result[i, j + 2] = np.max(boxes_corner[i, :, j])
    return result
