import numpy as np
from scipy.spatial.transform import Rotation


def transform(tf, points):
    r'''
    tf: transform matrix in 4x4
    points: in shape NxD, D>=3
    '''
    assert len(points.shape) >= 2 and points.shape[1] >= 3

    cord = points[..., :3]
    feat = points[..., 3:]
    cord = cord @ tf[:3, :3].T + tf[:3, 3]

    return np.concatenate([cord, feat], axis=-1)


def compose(tf0, tf1):
    return np.matmul(tf0, tf1)


def between(tf0, tf1):
    return compose(inverse(tf0), tf1)


def inverse(tf):
    inversed = tf.copy()
    inversed[:3, :3] = inversed[:3, :3].transpose()
    inversed[:3, 3] = -np.matmul(inversed[:3, :3], inversed[:3, 3])
    return inversed


def from_coeffs(coeffs):
    r'''
    transform matrix from coeffs [x,y,z, w,x,y,z]
    '''
    px = coeffs[0]
    py = coeffs[1]
    pz = coeffs[2]
    qw = coeffs[3]
    qx = coeffs[4]
    qy = coeffs[5]
    qz = coeffs[6]

    rotation = Rotation.from_quat(np.array([qx, qy, qz, qw], dtype=np.float32))
    translation = np.array([px, py, pz], dtype=np.float32)

    transform = np.identity(4, dtype=np.float32)
    transform[:3, :3] = rotation.as_matrix()
    transform[:3, 3] = translation

    return transform
