
from matplotlib import pyplot as plt
import numpy as np
import mdet.utils.io as io
from mdet.data.transforms.gt_sampler import FilterByNumpoints
from tqdm import tqdm
import sys

if int(sys.argv[1]) == 0:
    #  db_info_path = '/data/tmp/waymo/training_info_gt.pkl'
    db_info_path = '/data/ld00/waymo/det3d/training_info_gt.pkl'

    print(f'loading {db_info_path}')
    db_infos = io.load(db_info_path)
    print(f'{db_info_path} loaded')

    filter = FilterByNumpoints(100)

    print(f'filtering')
    car_infos = filter(db_infos[1], None, None)

    car_boxes = []
    for info in tqdm(car_infos):
        car_boxes.append(info['box'])
    car_boxes = np.array(car_boxes)

    car_size= car_boxes[:, 3:6]*2
    np.save('car_size.npy', car_size)
else:
    car_size = np.load('car_size.npy')

fig,  ax = plt.subplots(2,2)
ax[0,0].hist(car_size[:,0], bins=300)
ax[0,0].set_title('length')
ax[0,1].hist(car_size[:,1], bins=300)
ax[0,1].set_title('width')
ax[1,0].hist(car_size[:,2], bins=300)
ax[1,0].set_title('height')
ax[1,1].hist(car_size[:,0] * car_size[:,1] * car_size[:,2], bins=300)
ax[1,1].set_title('volume')
plt.show()
