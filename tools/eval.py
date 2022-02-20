import argparse
from functools import partial
import os
import os.path as osp

import pytorch_lightning as pl
from pytorch_lightning.profiler import PyTorchProfiler
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
    parser.add_argument('--split', type=str, default='eval',
                        choices=['train', 'eval'],  help='split to evaluate')
    parser.add_argument('--evaluate', action='store_true',
                        help='wether do evaluation')
    parser.add_argument('--evaluate_loss', action='store_true',
                        help='wether do evaluation loss')
    parser.add_argument(
        '--store_pred', help='output folder to store predictions(after postprocess)')
    parser.add_argument(
        '--store_formatted', help='output folder to store formatted predictions(after postprocess)')
    parser.add_argument(
        '--store_output', help='output folder to store raw predictions(before postprocess)')
    parser.add_argument('--show_output', action='store_true',
                        help='whether show model output')
    parser.add_argument('--show_pred', action='store_true',
                        help='whether show model prediction')
    parser.add_argument('--show_output_args', type=partial(yaml.load,
                        Loader=yaml.FullLoader), default='{}', help='show output args')
    parser.add_argument('--show_pred_args', type=partial(yaml.load,
                        Loader=yaml.FullLoader), default='{}', help='show model prediction')
    parser.add_argument('--gpu', type=int, nargs='+',
                        default=[0], help='specify the gpus used for training')
    parser.add_argument('--amp', default=False, action='store_true',
                        help='wether to do mix-precision training')
    parser.add_argument('--overfit', type=int, default=0,
                        help='overfit batch num, used for debug')
    parser.add_argument('--profile', default=False, action='store_true',
                        help='wether don profile, use together with overfit')
    parser.add_argument('--relax', default=False, action='store_true',
                        help='wether load checkpoint strictly')
    parser.add_argument('--dataset_root', type=str,
                        help='the dataset root folder, this will override config')
    parser.add_argument('--batch_size', type=int,
                        help='override batch in config')
    parser.add_argument('--trt_engine', type=str, help='tensor rt engine path')
    parser.add_argument('--trt_plugin', type=str, help='tensor rt plugin path')
    return parser.parse_args()


def main(args):
    config_name, _ = os.path.splitext(os.path.basename(args.config))
    print(f'Using config: {config_name}')
    print(f'Using gpu: {args.gpu}')

    # pre config load override
    if args.dataset_root:
        print(f'INFO: override dataset_root to {args.dataset_root}')
        GCFG['dataset_root'] = args.dataset_root

    if args.batch_size:
        print(f'INFO: override batch_size to {args.batch_size}')
        GCFG['batch_size'] = args.batch_size

    # hack config for evaluation
    config = ConfigLoader.load(args.config)
    config['data'][args.split]['shuffle'] = False
    config['runtime']['eval']['log_loss'] = args.evaluate_loss
    config['runtime']['eval']['evaluate'] = args.evaluate
    config['runtime']['eval']['evaluate_min_epoch'] = 0
    config['runtime']['eval']['formatted_path'] = args.store_formatted

    if args.trt_engine is not None:
        config['trt_engine'] = args.trt_engine
        config['trt_plugin'] = args.trt_plugin
        print(f'INFO: evaluating in TRT mode!')

    interest_set = set()
    epoch_interest_set = set()
    if (args.store_pred is not None) or (args.store_formatted is not None):
        # interest_set must be super set of epoch_interest_set
        interest_set |= {'pred', 'meta'}
        epoch_interest_set |= {'pred', 'meta'}
    if args.store_output is not None:
        # interest_set must be super set of epoch_interest_set
        interest_set |= {'output', 'meta'}
        epoch_interest_set |= {'output', 'meta'}
    if args.evaluate:
        # interest_set must be super set of epoch_interest_set
        interest_set |= {'pred', 'anno', 'meta'}
        epoch_interest_set |= {'anno', 'pred', 'meta'}
    if args.show_output:
        interest_set |= {'output', 'gt', 'input', 'meta'}
        print(f'show_output_args:\n{args.show_output_args}')
    if args.show_pred:
        interest_set |= {'pred', 'anno', 'data', 'meta'}
        print(f'show_pred_args:\n{args.show_pred_args}')

    def step_hook(sample, module):
        if args.show_output:
            sample.to('cpu')
            module.eval_codec.plot(sample, **args.show_output_args)

        if args.show_pred:
            module.eval_dataset.plot(sample, **args.show_pred_args)

    def epoch_hook(sample_list, module):
        if args.store_output is not None:
            os.makedirs(args.store_output, exist_ok=True)
            for sample in sample_list:
                sample_name = sample['meta']['sample_name']
                io.dump(sample['output'],
                        f'{args.store_output}/{sample_name}.pkl')

        if args.store_pred is not None:
            os.makedirs(args.store_pred, exist_ok=True)
            for sample in sample_list:
                sample_name = sample['meta']['sample_name']
                io.dump(sample['pred'],
                        f'{args.store_output}/{sample_name}.pkl')

    config['runtime']['eval']['interest_set'] = interest_set
    config['runtime']['eval']['epoch_interest_set'] = epoch_interest_set
    config['runtime']['eval']['step_hook'] = step_hook
    config['runtime']['eval']['epoch_hook'] = epoch_hook

    # NOTE: hack for none cleanly exits
    if args.show_output or args.show_pred:
        config['data'][args.split]['num_workers'] = 0

    # create lightning module
    if args.ckpt:
        pl_module = PlWrapper.load_from_checkpoint(
            config=config, checkpoint_path=args.ckpt, strict=(not args.relax))
    else:
        pl_module = PlWrapper(config=config)

    prof = PyTorchProfiler(
        filename='prof'
    )

    # profiler
    checkpoint_folder = './'
    if args.ckpt is not None:
        assert(osp.exists(args.ckpt) and osp.isfile(args.ckpt))
        checkpoint_folder = osp.dirname(args.ckpt)

    profiler = None
    if args.profile:
        profiler = PyTorchProfiler(
            dirpath=checkpoint_folder,  # use same as checkpoint
            filename='profile',
            schedule=torch.profiler.schedule(
                wait=2, warmup=2, active=6, repeat=1),
        )

    # setup trainner
    trainer = pl.Trainer(
        logger=False,
        gpus=args.gpu,
        sync_batchnorm=len(args.gpu) > 1,
        strategy='ddp' if len(args.gpu) > 1 else None,
        overfit_batches=args.overfit,
        profiler=profiler,
        precision=16 if args.amp else 32,
    )

    # do evaluation
    loader_name = 'val_dataloader' if args.split == 'eval' else f'{args.split}_dataloader'
    trainer.validate(pl_module, dataloaders=getattr(pl_module, loader_name)())


if __name__ == '__main__':
    main(parse_args())
