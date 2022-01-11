import argparse
import os
import os.path as osp

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning.loggers as loggers
from pytorch_lightning.profiler import PyTorchProfiler
import torch

import mdet.data
import mdet.model
import mdet.utils.config_loader as ConfigLoader
import mdet.utils.numpy_pickle
from mdet.utils.pl_wrapper import PlWrapper


r'''
Train model
'''


def parse_args():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('config', help='training config file')
    parser.add_argument(
        '--ckpt', help='the checkpoint file to resume from')
    parser.add_argument(
        '--work_dir', default='./log', help='the folder logs store')
    parser.add_argument('--gpu', type=int, nargs='+',
                        default=[0], help='specify the gpus used for training')
    parser.add_argument('--amp', default=False, action='store_true',
                        help='wether to do mix-precision training')
    parser.add_argument('--overfit', type=int, default=0,
                        help='overfit batch num, used for debug')
    parser.add_argument('--profile', default=False, action='store_true',
                        help='wether don profile, use together with overfit')
    parser.add_argument('--lr_scale', type=float,
                        help='when batch size varies, we need to rescale learning rate')
    return parser.parse_args()


def main(args):
    config_name, _ = os.path.splitext(os.path.basename(args.config))
    print(f'Using config: {config_name}')
    print(f'Using gpu: {args.gpu}')

    # load config
    config = ConfigLoader.load(args.config)

    # config override
    if args.lr_scale is not None:
        config['lr_scale'] = args.lr_scale

    # lightning module
    pl_module = PlWrapper(config)

    max_epochs = config['fit']['max_epochs']
    print(f'Total epochs: {max_epochs}')

    # recovery version from checkpoint path
    version = None
    if args.ckpt is not None:
        assert(osp.exists(args.ckpt) and osp.isfile(args.ckpt))
        version_str = osp.basename(osp.dirname(args.ckpt))
        if version_str.startswith('version_'):
            version = int(version_str.split('_')[1])

    # setup loggers
    logger_list = []
    for logger_cfg in config['runtime']['train']['logger']:
        cfg = logger_cfg.copy()
        type = cfg.pop('type')
        logger_list.append(loggers.__dict__[type](
            args.work_dir, name=config_name, version=version, **cfg))
        if version is None:
            version = logger_list[-1].version

    # get experiment_version from first logger
    experiment_version = logger_list[0].version
    if not isinstance(experiment_version, str):
        experiment_version = f'version_{experiment_version}'

    # callbacks
    callbacks = []
    # make checkpoint path identical with log
    checkpoint_folder = osp.join(
        args.work_dir, config_name, experiment_version)
    callbacks.append(ModelCheckpoint(dirpath=checkpoint_folder))

    # profiler
    profiler = None
    if args.profile:
        profiler = PyTorchProfiler(
            dirpath=checkpoint_folder,  # use same as checkpoint
            filename='profile',
            schedule=torch.profiler.schedule(
                wait=2, warmup=2, active=6, repeat=1),
            record_shapes=True,
        )

    # grad clip
    grad_clip_val = None
    grad_clip_alg = None
    if 'grad_clip' in config['fit']:
        grad_clip_val = config['fit']['grad_clip']['value']
        grad_clip_alg = config['fit']['grad_clip']['type']

    # setup trainner
    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        logger=logger_list,
        callbacks=callbacks,
        max_epochs=max_epochs,
        gpus=args.gpu,
        sync_batchnorm=len(args.gpu) > 1,
        strategy='ddp' if len(args.gpu) > 1 else None,
        overfit_batches=args.overfit,
        profiler=profiler,
        gradient_clip_algorithm=grad_clip_alg,
        gradient_clip_val=grad_clip_val,
        precision=16 if args.amp else 32,
    )

    # do fit
    trainer.fit(pl_module, ckpt_path=args.ckpt)


if __name__ == '__main__':
    main(parse_args())
