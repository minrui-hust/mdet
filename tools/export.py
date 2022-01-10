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
    parser.add_argument('--output', type=str, help='the output onnx model path')
    parser.add_argument('--gpu', type=int, nargs='+',
                        default=[0], help='specify the gpus used for training')
    parser.add_argument('--relax', default=False, action='store_true',
                        help='wether load checkpoint strictly')
    return parser.parse_args()


def main(args):
    config_name, _ = os.path.splitext(os.path.basename(args.config))
    print(f'Using config: {config_name}')
    print(f'Using gpu: {args.gpu}')

    # hack infer config for export
    config = ConfigLoader.load(args.config)
    config['data']['infer']['shuffle'] = False
    config['data']['infer']['pin_memory'] = False
    config['data']['infer']['num_workers'] = 1
    config['data']['infer']['batch_size'] = 1

    # create lightning module
    if args.ckpt:
        pl_module = PlWrapper.load_from_checkpoint(config=config, checkpoint_path=args.ckpt, strict=(not args.relax))
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

    pl_module.export(output_file=output_file)


if __name__ == '__main__':
    main(parse_args())
