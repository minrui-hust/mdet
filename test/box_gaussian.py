import numpy as np
from mdet.utils.gaussian import box_gaussian_kernel_2D
from matplotlib import pyplot as plt

box = np.array([0, 0, 0, 2.5, 1.0, 0.9, 0.866, 0.5], dtype=np.float32)

h = box_gaussian_kernel_2D(box, 0.2, 2)
print(h.shape)

plt.axis('equal')
plt.pcolormesh(np.arange(h.shape[1]+1), np.arange(h.shape[0]+1), h, cmap='hot')
plt.show()
