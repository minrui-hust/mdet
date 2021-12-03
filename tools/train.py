import argparse
from tools.runner import Runner
from utils.config_loader import ConfigLoader
import pytorch_lightning as pl


def parse_args():
    parser = argparse.ArgumentParser(description='Train a module')
    return parser


def main():
    args = parse_args()
    config = ConfigLoader.load(args.config)
    runner = Runner(config)
    trainer = pl.Trainer()
    trainer.fit(runner)
