import os
import os.path as osp
import argparse
from mdet.utils.pl_wrapper import PlWrapper
import mdet.utils.config_loader as ConfigLoader
import mdet.data
import mdet.model
import pytorch_lightning as pl
import pytorch_lightning.loggers as loggers
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

r'''
Test model on test set.
Optionally save predict output and formated output(for submission)
'''


def parse_args():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('config', help='training config file')
    parser.add_argument('--ckpt', help='the checkpoint file to resume from')
    parser.add_argument('--output', help='output folder to store predictions')
    parser.add_argument('--format', help='path to store formated predictions')
    parser.add_argument('--gpu', type=int, nargs='+',
                        default=[0], help='specify the gpus used for training')
    return parser.parse_args()


def main(args):
    config_name, _ = os.path.splitext(os.path.basename(args.config))
    print(f'Using config: {config_name}')
    print(f'Using checkpoint: {args.ckpt}')
    print(f'Using gpu: {args.gpu}')

    # create lightning module
    config = ConfigLoader.load(args.config)
    config['runtime']['test']['output_folder'] = args.output
    config['runtime']['test']['format_path'] = args.format
    pl_module = PlWrapper(config)

    # setup trainner
    trainer = pl.Trainer(
        logger=False,
        gpus=args.gpu,
        sync_batchnorm=len(args.gpu) > 1,
        strategy='ddp' if len(args.gpu) > 1 else None,
        overfit_batches=100,
    )

    # do fit
    trainer.test(pl_module, ckpt_path=args.ckpt)


if __name__ == '__main__':
    main(parse_args())
