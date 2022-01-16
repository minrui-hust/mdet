import argparse
from functools import partial

import yaml

import mdet.data.datasets
from mdet.utils.factory import FI


def parse_args():
    parser = argparse.ArgumentParser(description='Dataset converter')
    parser.add_argument('dataset', help='name of dataset type')
    parser.add_argument('--root_path',
                        type=str,
                        help='root path of the raw dataset')
    parser.add_argument('--splits',
                        type=str,
                        nargs='+',
                        default=['train', 'val', 'test'],
                        help='')
    parser.add_argument('--create_args',
                        type=partial(yaml.load, Loader=yaml.FullLoader),
                        default='{}',
                        help='show output args')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    creator_name = f'{args.dataset}GtDatabaseCreator'
    creator = FI.create(dict(type=creator_name))

    print(f'create args:\n {args.create_args}')

    for split in args.splits:
        print(f'Processing split: {split}')
        creator(args.root_path, split, **args.create_args)


if __name__ == '__main__':
    main()
