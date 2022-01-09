import argparse
import mdet.data.datasets.waymo_det3d.converter as waymo_converter


def parse_args():
    parser = argparse.ArgumentParser(description='Dataset converter')
    parser.add_argument('dataset', help='name of dataset type')
    parser.add_argument('--root_path', type=str,
                        help='root path of the raw dataset')
    parser.add_argument('--out_path', type=str, help='')
    parser.add_argument('--splits', type=str, nargs='+',
                        default=['train', 'val', 'test'], help='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    converter_name = f'{args.dataset}_converter'
    converter = globals()[converter_name]

    for split in args.splits:
        print(f'Processing split: {split}')
        converter.convert(args.root_path, args.out_path, split)


if __name__ == '__main__':
    main()
