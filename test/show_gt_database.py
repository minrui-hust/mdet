from mdet.utils.factory import FI
import mdet.data
import mdet.utils.io as io
import numpy as np
import argparse

import open3d as o3d
import os

from mdet.utils.viz import Visualizer


def parse_args():
    parser = argparse.ArgumentParser(description='Dataset converter')
    parser.add_argument('dataset', help='name of dataset type')
    parser.add_argument('--root_path',
                        type=str,
                        help='root path of the raw dataset')
    parser.add_argument('--splits',
                        type=str,
                        nargs='+',
                        default=['train', 'val', 'test'],
                        help='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    root_path = args.root_path
    split = args.splits[0]

    gt_info_path = os.path.join(root_path, f'{split}_info_gt.pkl')

    gt_info_dict = io.load(gt_info_path)

    for info in gt_info_dict[1]:
        pcd_path = info['sweeps'][0]['pcd_path']
        if pcd_path is not None:
            pcd = io.load(pcd_path, compress=True)
            box = info['box']

            print(info['seq_id'], info['frame_id'], info['name'])
            print(pcd.shape)

            vis = Visualizer()
            vis.add_points(pcd)
            vis.add_box(box[np.newaxis, :])
            vis.show()


if __name__ == '__main__':
    main()
