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
    def __init__(self, grid=0.2, min_radius=2, eps=1e-4):
        super().__init__()
        self.grid = grid
        self.min_radius = min_radius
        self.eps = eps

    def __call__(self, heatmap, box, center):
        # calc gaussian kernel parameters
        rotation = box[6:8]

        extend_x = max(box[3], (self.min_radius+0.5)*self.grid)
        extend_y = max(box[4], (self.min_radius+0.5)*self.grid)
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
        kernel = gaussian_kernel_2D(
            [rx, ry], [sigma_x, sigma_y], rotation, self.eps)

        # draw kernel onto heatmap
        draw_gaussian_kernel_2D(heatmap, center, kernel)
