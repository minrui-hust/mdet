import argparse
from functools import partial
import os
import os.path as osp

import pytorch_lightning as pl
import torch
import yaml

import mdet.data
import mdet.model
import mdet.utils.config_loader as ConfigLoader
from mdet.utils.global_config import GCFG
import mdet.utils.io as io
import mdet.utils.numpy_pickle
from mdet.utils.pl_wrapper import PlWrapper


r'''
Evaluate model on evaluation set.
Optionally save predict output and evaluation metric
'''


def parse_args():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('config', help='training config file')
    parser.add_argument('--ckpt', help='the checkpoint file to resume from')
    parser.add_argument('--output', type=str,
                        help='the output onnx model path')
    parser.add_argument('--gpu', type=int, nargs='+',
                        default=[0], help='specify the gpus used for training')
    parser.add_argument('--relax', default=False, action='store_true',
                        help='wether load checkpoint strictly')
    parser.add_argument('--dataset_root', type=str,
                        help='the dataset root folder, this will override config')
    parser.add_argument(
        '--type', type=str, choices=['onnx', 'torch'], default='onnx', help='export type')
    return parser.parse_args()


def main(args):
    config_name, _ = os.path.splitext(os.path.basename(args.config))
    print(f'Using config: {config_name}')
    print(f'Using gpu: {args.gpu}')

    # pre config load override
    if args.dataset_root:
        print(f'INFO: override dataset_root to {args.dataset_root}')
        GCFG['dataset_root'] = args.dataset_root

    # hack infer config for export
    config = ConfigLoader.load(args.config)
    config['data']['infer']['shuffle'] = False
    config['data']['infer']['pin_memory'] = False
    config['data']['infer']['num_workers'] = 1
    config['data']['infer']['batch_size'] = 1

    # create lightning module
    if args.ckpt:
        pl_module = PlWrapper.load_from_checkpoint(
            config=config, checkpoint_path=args.ckpt, strict=(not args.relax))
    else:
        pl_module = PlWrapper(config=config)

    # setup output folder
    if not args.output:
        output_folder = ""
        if args.ckpt is not None:
            ckpt_folder = osp.dirname(args.ckpt)
            version_str = osp.basename(ckpt_folder)
            if version_str.startswith('version_'):
                output_folder = ckpt_folder
        output_file = osp.join(output_folder, 'model.onnx')
    else:
        output_file = args.output

    if args.type == 'onnx':
        pl_module.export_onnx(output_file=output_file)
    elif args.type == 'torch':
        pl_module.export_torch(output_file=output_file)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main(parse_args())
