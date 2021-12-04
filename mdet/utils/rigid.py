import numpy as np


def transform(tf, points):
    r'''
    tf: transform matrix in 4x4
    points: in shape NxD, D>=3
    '''
    assert len(points.shape) == 2 and points.shape[1] >= 3

    cord = points[:, :3]
    feat = points[:, 3:]
    cord = np.matmul(tf[:3, :3], cord.transpose()).transpose() + tf[:3, 3]

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
