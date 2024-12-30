import logging
from pathlib import Path
from argparse import ArgumentParser
from pipeline import SurfacePipeline

parser = ArgumentParser(description='extract a surface patch of a sample region and save it')
parser.add_argument('-s', '--source-path',
                    type=str,
                    required=True,
                    help='path to the source files (directory containing C0001.vtk)')
parser.add_argument('-p', '--parameter-path',
                    type=str,
                    required=True,
                    help='path to the sample parameter files (directory containing C0001.pvsm)')
parser.add_argument('-o', '--output-path',
                    type=str,
                    required=True,
                    help='path to save new surface geometry files')


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    state_path = Path(args.parameter_path)
    source_path = Path(args.source_path)
    output_path = Path(args.output_path)
    assert state_path.is_dir(), '%s is not a directory' % state_path.as_posix()
    assert source_path.is_dir(), '%s is not a directory' % source_path.as_posix()
    assert output_path.is_dir(), '%s is not a directory' % output_path.as_posix()
    for state in state_path.iterdir():
        logging.info(' processing %s', state.stem)
        pipeline = SurfacePipeline()
        name = pipeline.set_state(state)
        source = source_path.joinpath(name)
        assert source.is_file(), '%s is not a file' % source.as_posix()
        pipeline.set_source(source)
        pipeline.write(output_path.joinpath(state.stem + '_surface.vtk'))


if __name__ == '__main__':
    main()
