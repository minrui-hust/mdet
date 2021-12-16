import torch
import argparse
from mdet.utils.runner import Runner
import mdet.utils.config_loader as ConfigLoader
import mdet.data
import mdet.model


def parse_args():
    parser = argparse.ArgumentParser(description='Train a module')
    parser.add_argument('config_path')
    return parser.parse_args()


def main():
    args = parse_args()
    config = ConfigLoader.load(args.config_path)
    runner = Runner(config)
    optim, sched = runner.configure_optimizers()
    print(optim)
    print(sched)


if __name__ == '__main__':
    main()
