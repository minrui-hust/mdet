import argparse
from functools import partial
import os
import os.path as osp

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning.loggers as loggers
from pytorch_lightning.profiler import PyTorchProfiler
import torch
from torch.profiler.profiler import tensorboard_trace_handler
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
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val'],  help='split to evaluate')
    parser.add_argument('--evaluate', action='store_true',
                        help='wether do evaluation')
    parser.add_argument(
        '--store_pred', help='output folder to store predictions(after postprocess)')
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
    parser.add_argument('--overfit', type=int, default=0,
                        help='overfit batch num, used for debug')
    parser.add_argument('--profile', default=False, action='store_true',
                        help='wether don profile, use together with overfit')
    return parser.parse_args()


def main(args):
    config_name, _ = os.path.splitext(os.path.basename(args.config))
    print(f'Using config: {config_name}')
    print(f'Using gpu: {args.gpu}')

    # hack config for evaluation
    config = ConfigLoader.load(args.config)
    config['data'][args.split]['shuffle'] = False
    config['runtime']['val']['log_loss'] = False
    config['runtime']['val']['evaluate'] = args.evaluate

    interest_set = set()
    epoch_interest_set = set()
    if args.store_pred is not None:
        interest_set |= {'pred', 'meta'}
        epoch_interest_set |= {'pred', 'meta'}
    if args.store_output is not None:
        interest_set |= {'output', 'meta'}
        epoch_interest_set |= {'ourput', 'meta'}
    if args.show_output:
        interest_set |= {'output', 'gt', 'input', 'meta'}
        print(f'show_output_args:\n{args.show_output_args}')
    if args.show_pred:
        interest_set |= {'pred', 'anno', 'data', 'meta'}
        print(f'show_pred_args:\n{args.show_pred_args}')

    def step_hook(sample, module):
        #  sample.to('cpu')
        if args.show_output:
            module.eval_codec.plot(sample, **args.show_output_args)

        if args.show_pred:
            module.val_dataset.plot(sample, **args.show_pred_args)

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

    config['runtime']['val']['interest_set'] = interest_set
    config['runtime']['val']['epoch_interest_set'] = epoch_interest_set
    config['runtime']['val']['step_hook'] = step_hook
    config['runtime']['val']['epoch_hook'] = epoch_hook

    # NOTE: hack for none cleanly exits
    if args.show_output or args.show_pred:
        config['data'][args.split]['num_workers'] = 0

    # create lightning module
    pl_module = PlWrapper(config)

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
        profiler=profiler
    )

    # do validation
    trainer.validate(pl_module,
                     ckpt_path=args.ckpt,
                     dataloaders=getattr(
                         pl_module, f'{args.split}_dataloader')()
                     )


if __name__ == '__main__':
    main(parse_args())
