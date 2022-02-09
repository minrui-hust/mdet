import numpy as np
from mdet.core.geometry2d import rotate_points
import time

points = np.random.rand(300000, 2).astype(np.float32)
angle = np.random.rand(300000).astype(np.float32)
rot = np.stack([np.cos(angle), np.sin(angle)], axis=-1)

tick = time.time()
res = rotate_points(points, rot)
print(time.time()-tick)
