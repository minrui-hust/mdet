import argparse
import os
import os.path as osp

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning.loggers as loggers
from pytorch_lightning.profiler import PyTorchProfiler

import mdet.data
import mdet.model
import mdet.utils.config_loader as ConfigLoader
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
        '--workspace', default='./workspace', help='the workspace folder output logs store')
    parser.add_argument('--gpu', type=int, nargs='+',
                        default=[0], help='specify the gpus used for training')
    return parser.parse_args()


def main(args):
    config_name, _ = os.path.splitext(os.path.basename(args.config))
    print(f'Using config: {config_name}')
    print(f'Using gpu: {args.gpu}')

    # lightning module
    config = ConfigLoader.load(args.config)
    pl_module = PlWrapper(config)

    max_epochs = config['runtime']['train']['max_epochs']
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
            osp.join(args.workspace, 'log'), name=config_name, version=version, **cfg))
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
        args.workspace, 'checkpoint', config_name, experiment_version)
    callbacks.append(ModelCheckpoint(dirpath=checkpoint_folder))

    prof = PyTorchProfiler(
        filename='prof'
    )

    # setup trainner
    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        logger=logger_list,
        callbacks=callbacks,
        max_epochs=max_epochs,
        gpus=args.gpu,
        sync_batchnorm=len(args.gpu) > 1,
        strategy='ddp' if len(args.gpu) > 1 else None,
        overfit_batches=100,
        profiler=prof,
    )

    # do fit
    trainer.fit(pl_module, ckpt_path=args.ckpt)


if __name__ == '__main__':
    main(parse_args())
