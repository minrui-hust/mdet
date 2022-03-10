import math

import numpy as np

from mdet.core.box_np_ops import corners_nd, rotate2d
from mdet.utils.factory import FI
from mdet.utils.gaussian import (
    draw_gaussian,
    draw_gaussian_kernel_2D,
    gaussian_kernel_2D,
    gaussian_radius,
)


@FI.register
class NaiveGaussianBoxHeatmapEncoder(object):
    def __init__(self, grid=0.2, min_radius=2, min_overlap=0.1):
        super().__init__()
        self.grid = grid
        self.min_radius = min_radius
        self.min_overlap = min_overlap

    def __call__(self, heatmap, box, center):
        l, w = box[3] / self.grid, box[4] / self.grid
        radius = gaussian_radius((l, w), self.min_overlap)
        radius = max(self.min_radius, int(radius))
        draw_gaussian(heatmap, center, radius)


@FI.register
class GaussianBoxHeatmapEncoder(object):
    def __init__(self, grid=0.2, min_radius=2, offset_enable=False, ratio=[1.0, 1.0], eps=1e-4):
        super().__init__()
        self.grid = grid
        self.min_radius = min_radius
        self.offset_enable = offset_enable
        self.ratio = ratio
        self.eps = eps

    def __call__(self, heatmap, box, center):
        # calc gaussian kernel parameters
        rotation = box[6:8]

        extend_x = max(box[3]*self.ratio[0], (self.min_radius+0.5)*self.grid)
        extend_y = max(box[4]*self.ratio[1], (self.min_radius+0.5)*self.grid)
        extend = np.array([extend_x, extend_y], dtype=np.float32)

        # corners
        corners = rotate2d(corners_nd(
            extend[np.newaxis, ...]), rotation[np.newaxis, ...]).squeeze(0)

        rx, ry = np.max(corners, axis=0)
        rx = math.floor(rx/self.grid + 0.5)
        ry = math.floor(ry/self.grid + 0.5)

        sigma_x = extend_x/self.grid/3
        sigma_y = extend_y/self.grid/3

        # calc kernel
        offset = 0
        if self.offset_enable:
            offset = center-(np.floor(center) + 0.5)
        kernel = gaussian_kernel_2D(
            [rx, ry], [sigma_x, sigma_y], rotation, offset=offset, eps=self.eps)

        # draw kernel onto heatmap
        draw_gaussian_kernel_2D(heatmap, np.floor(center), kernel)


@FI.register
class GaussianBoxKeypointEncoder(object):
    def __init__(self, grid=0.2, min_radius=1, offset_enable=False, ratio=[1.0, 1.0], eps=1e-4):
        super().__init__()
        self.grid = grid
        self.min_radius = min_radius
        self.offset_enable = offset_enable
        self.ratio = ratio
        self.eps = eps

    def __call__(self, heatmap, box, keypoints):
        # calc gaussian kernel parameters
        rotation = box[6:8]

        # effective extend only box's 1/3
        extend_x = min(box[3]*self.ratio[0], box[4]*self.ratio[0])
        extend_y = min(box[3]*self.ratio[1], box[4]*self.ratio[1])

        extend_x = max(extend_x, (self.min_radius+0.5)*self.grid)
        extend_y = max(extend_y, (self.min_radius+0.5)*self.grid)
        extend = np.array([extend_x, extend_y], dtype=np.float32)

        # corners
        corners = rotate2d(corners_nd(
            extend[np.newaxis, ...]), rotation[np.newaxis, ...]).squeeze(0)

        rx, ry = np.max(corners, axis=0)
        rx = math.floor(rx/self.grid + 0.5)
        ry = math.floor(ry/self.grid + 0.5)

        sigma_x = extend_x/self.grid/3
        sigma_y = extend_y/self.grid/3

        for keypoint in keypoints:
            # calc kernel
            offset = 0
            if self.offset_enable:
                offset = keypoint-(np.floor(keypoint) + 0.5)
            kernel = gaussian_kernel_2D(
                [rx, ry], [sigma_x, sigma_y], rotation, offset=offset, eps=self.eps)

            # draw kernel onto heatmap
            draw_gaussian_kernel_2D(heatmap, np.floor(keypoint), kernel)
