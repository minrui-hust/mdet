import numpy as np


def gaussian_radius(det_size, min_overlap=0.5):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius +
                               bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian_kernel_2D(shape, sigma, rot, offset=0.0, eps=1e-6):
    r'''
    Args:
        shape: int array like with length 2, [x_radius, y_radius]
        sigma: std variance, array like with length 2
        rotation: rotation matrix of 2x2
    '''
    # cordinates
    rx, ry = shape[0], shape[1]
    y, x = np.broadcast_arrays(*np.ogrid[-ry:ry+1, -rx:rx+1])
    cord = np.stack([x, y], axis=-1) - offset  # (2*ry+1)x(2*rx+1)x2

    # the rotation matrix
    rotm = np.array([[rot[0], rot[1]], [-rot[1], rot[0]]], dtype=np.float32)

    # information matrix
    H = rotm.T @ np.diag([1/(s*s) for s in sigma]) @ rotm

    # mahalanobis distance
    D = (cord[:, :, np.newaxis, :] @ H @
         cord[:, :, :, np.newaxis]).squeeze(-1).squeeze(-1)

    # kernel value
    kernel = np.exp(-0.5*D)

    # filter value too small
    kernel[kernel < eps] = 0

    return kernel


def draw_gaussian_kernel_2D(heatmap, center, kernel):
    r'''
    Args:
        heatmap: heatmap to draw on, shape NxN
        center: center of kernel on heatmap
        kernel: the kernel to draw
    '''
    x, y = int(center[0]), int(center[1])

    h_heatmap, w_heatmap = heatmap.shape[0:2]
    h_kernel, w_kernel = kernel.shape[0:2]
    ry_kernel, rx_kernel = int((h_kernel-1)/2), int((w_kernel-1)/2)

    left, right = min(x, rx_kernel), min(w_heatmap - x, rx_kernel + 1)
    top, bottom = min(y, ry_kernel), min(h_heatmap - y, ry_kernel + 1)

    masked_heatmap = heatmap[y - top:y + bottom,
                             x - left:x + right]
    masked_gaussian = kernel[ry_kernel - top: ry_kernel + bottom,
                             rx_kernel - left: rx_kernel + right]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)

    return heatmap
