import os
import argparse
import logging
from pathlib import Path

from mesh import MeshGenerator


parser = argparse.ArgumentParser(description='create a parametrized mesh')
parser.add_argument('-r', '--resolution', type=int, default=16,
                    help='Resolution of the mesh in each dimension')
parser.add_argument('-t', '--type', type=str, default='BOX',
                    choices=['BOX', 'SPHRERE', 'CYLINDER'], help='Type of mesh')
parser.add_argument('-s', '--save_path', type=str, required=True,
                    help='save location (if nothing is specified the current working directory is used)')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    save_dir = Path(os.getcwd())
    args = parser.parse_args()
    if args.save_path is not None:
        save_dir = Path(args.save_path)
        if not save_dir.exists():
            raise NotADirectoryError(f'{save_dir} is not a directory')
    save_dir = save_dir.joinpath(args.type)
    if not save_dir.exists():
        os.mkdir(save_dir)
    MeshGenerator.create(MeshGenerator.Type(args.type))
    MeshGenerator.generate_mesh(args.resolution)
    MeshGenerator.save(save_dir)
