import logging
from pathlib import Path
from argparse import ArgumentParser
from timeit import default_timer

from pipeline import GeometryPipeline

parser = ArgumentParser(prog='calculate geometric priors',
                        description='add geometric priors to a surface file')
parser.add_argument('-s', '--surface-path',
                    type=str,
                    required=True,
                    help='path to the surface files (directory containing C0001_surface.vtk)')
parser.add_argument('-o', '--output-path',
                    type=str,
                    required=True,
                    help='path to save new surface files')


def main() -> None:
    logging.basicConfig(level=logging.DEBUG)
    args = parser.parse_args()
    surface_path = Path(args.surface_path)
    output_path = Path(args.output_path)
    assert surface_path.is_dir(), '%s is not a directory' % surface_path.as_posix()
    assert output_path.is_dir(), '%s is not a directory' % output_path.as_posix()
    for surface in surface_path.iterdir():
        logging.info(' processing %s', surface.stem)
        pipeline = GeometryPipeline()
        pipeline.set_source(surface)
        t0 = default_timer()
        pipeline.save_geometric_priors(output_path.joinpath(surface.name))
        t1 = default_timer()
        logging.info(' time: %fs', t1 - t0)


if __name__ == '__main__':
    main()
