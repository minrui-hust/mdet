import argparse
import mdet.data.datasets.waymo.summary as waymo_summary


def parse_args():
    parser = argparse.ArgumentParser(description='Dataset converter')
    parser.add_argument('dataset', help='name of dataset type')
    parser.add_argument('--root_path', type=str,
                        help='root path of the raw dataset')
    parser.add_argument('--splits', type=str, nargs='+',
                        default=['train', 'val', 'test'], help='')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    summary_name = f'{args.dataset}_summary'
    summary = globals()[summary_name]

    for split in args.splits:
        print(f'Processing split: {split}')
        summary.summary(args.root_path, split)


if __name__ == '__main__':
    main()
