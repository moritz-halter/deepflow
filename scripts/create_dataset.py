import argparse
import logging
from pathlib import Path

from data.datasets.aneurisk import create_datasets

parser = argparse.ArgumentParser(description='compile a dataset datasets')
parser.add_argument('-r', '--resolution',
                    type=int,
                    required=True,
                    help='the grid resolution of the dataset')
parser.add_argument('-o', '--output-path',
                    type=str,
                    required=True,
                    help='directory to save the dataset in')

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    args = parser.parse_args()
    output_path = Path(args.output_path)
    if not output_path.exists():
        output_path.mkdir()

    create_datasets(output_path, args.resolution)
