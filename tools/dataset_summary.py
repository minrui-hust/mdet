import argparse
import mdet.data.datasets
from mdet.utils.factory import FI
import yaml
from functools import partial


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
    parser.add_argument('--summary_args',
                        type=partial(yaml.load, Loader=yaml.FullLoader),
                        default='{}',
                        help='show output args')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    summary_name = f'{args.dataset}Summary'
    summary = FI.create(dict(type=summary_name))

    print(f'summary args:\n {args.summary_args}')

    for split in args.splits:
        print(f'Processing split: {split}')
        summary(args.root_path, split, **args.summary_args)


if __name__ == '__main__':
    main()
