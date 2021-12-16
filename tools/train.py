import os
import argparse
from mdet.utils.pl_wrapper import PlWrapper
import mdet.utils.config_loader as ConfigLoader
import mdet.data
import mdet.model
import pytorch_lightning as pl
import pytorch_lightning.loggers as loggers
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('config', help='training config file')
    parser.add_argument(
        '--ckpt', help='the checkpoint file to resume from')
    parser.add_argument(
        '--workspace', default='./log', help='the workspace folder output logs store')
    parser.add_argument('--gpu', type=int, nargs='+',
                        default=[0], help='specify the gpus used for training')
    return parser.parse_args()


def main():
    args = parse_args()
    config_name, _ = os.path.splitext(os.path.basename(args.config))
    print(f'Using config: {config_name}')
    print(f'Using gpu: {args.gpu}')

    # lightning module
    config = ConfigLoader.load(args.config)
    pl_module = PlWrapper(config)

    max_epochs = config['runtime']['max_epochs']
    print(f'Total epochs: {max_epochs}')

    # setup loggers
    logger_list = []
    for logger_cfg in config['runtime']['logging']['logger']:
        cfg = logger_cfg.copy()
        type = cfg.pop('type')
        logger_list.append(loggers.__dict__[type](
            args.workspace, name=config_name, **cfg))

    # callbacks
    callbacks = []
    # make checkpoint path identical with log
    callbacks.append(ModelCheckpoint(dirpath=args.workspace))

    # setup trainner
    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        logger=logger_list,
        callbacks=callbacks,
        max_epochs=max_epochs,
        gpus=args.gpu,
        sync_batchnorm=len(args.gpu) > 1,
        strategy='ddp' if len(args.gpu) > 1 else None,
    )

    # do fit
    trainer.fit(pl_module, ckpt_path=args.ckpt)


if __name__ == '__main__':
    main()
