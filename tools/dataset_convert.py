import argparse
from functools import partial

import yaml

import mdet.data.datasets
from mdet.utils.factory import FI


def parse_args():
    parser = argparse.ArgumentParser(description='Dataset converter')
    parser.add_argument('dataset', help='name of dataset type')
    parser.add_argument('--root_path', type=str,
                        help='root path of the raw dataset')
    parser.add_argument('--out_path', type=str, help='')
    parser.add_argument('--splits', type=str, nargs='+',
                        default=['train', 'val', 'test'], help='')
    parser.add_argument('--args', type=partial(yaml.load,
                        Loader=yaml.FullLoader), default='{}', help='additional args')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(f'additional args:\n{args.args}')

    converter_name = f'{args.dataset}Converter'
    converter = FI.create(dict(type=converter_name))

    for split in args.splits:
        print(f'Processing split: {split}')
        converter(args.root_path, args.out_path, split, **args.args)


if __name__ == '__main__':
    main()
