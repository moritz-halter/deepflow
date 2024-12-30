import logging
import re
import shutil
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
from scp import SCPClient
from paramiko import SSHClient

from pipeline import DataPipeline
from utils import concatenate_dicts


parser = ArgumentParser(description='preview the sampling for a single timestep')
parser.add_argument('-s', '--source-path',
                    type=str,
                    required=True,
                    help='path to the source files (directory containing C0001.vtk)')
parser.add_argument('-g', '--geometry-path',
                    type=str,
                    required=True,
                    help='path to the surface geometry file (directory containing C0001_surface.vtk)')
parser.add_argument('-p', '--parameter-path',
                    type=str,
                    required=True,
                    help='path to the sample parameter file (directory containing C0001.pvsm)')
parser.add_argument('-o', '--output-path',
                    type=str,
                    required=True,
                    help='directory to save new sample files in')
parser.add_argument('-m', '--mesh-path',
                    type=str,
                    help='path to the mesh file')
parser.add_argument('-t0', '--t-start', type=int,
                    help='first time index')
parser.add_argument('-t1', '--t-end', type=int,
                    help='last time index')


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    source_path = Path(args.source_path)
    geometry_path = Path(args.geometry_path)
    state_path = Path(args.parameter_path)
    output_path = Path(args.output_path)
    assert state_path.is_dir(), '%s is not a directory' % state_path.as_posix()
    assert geometry_path.is_dir(), '%s is not a directory' % geometry_path.as_posix()
    assert source_path.is_dir(), '%s is not a directory' % source_path.as_posix()
    assert output_path.is_dir(), '%s is not a directory' % output_path.as_posix()
    pipeline = DataPipeline()
    param_coords = None
    if args.mesh_path is not None:
        mesh_path = Path(args.mesh_path)
        assert Path(args.mesh_path).is_dir(), '%s is not a directory' % Path(args.mesh_path).as_posix()
        pipeline.set_mesh(mesh_path.joinpath('mesh.vtk'))
        param_coords = np.load(mesh_path.joinpath('param_coords.npy'))
    for state in state_path.iterdir():
        sample_name = state.stem
        if sample_name != 'C0069_1':
            continue
        if len(sample_name) == 7:
            sample_name = sample_name[:-2]
        ssh = SSHClient()
        ssh.load_system_host_keys()
        ssh.connect(hostname='HOST', # SET HOST AND CREDENTIAL HERE
                    username='USERNAME',
                    password='PASSWORD')
        scp = SCPClient(ssh.get_transport())
        local_path = source_path.joinpath(sample_name).joinpath('VTK/VTK')
        if not local_path.exists():
            local_path.mkdir(parents=True)
        scp.get(remote_path='PATH/TO/REMOTE/RAW/DATASET' + sample_name + '/VTK/VTK', # SET PATH HERE
                local_path=source_path.joinpath(sample_name).joinpath('VTK').as_posix(),
                recursive=True)
        logging.info(' processing %s', state.stem)
        name = pipeline.set_state(state)
        files_path = source_path.joinpath(name[:-4]).joinpath('VTK/VTK')
        surface = geometry_path.joinpath(state.stem + '_surface.vtk')
        assert files_path.is_dir(), '%s is not a directory' % files_path.as_posix()
        assert surface.is_file(), '%s is not a file' % surface.as_posix()
        files = sorted(list(files_path.iterdir()))
        dynamic = []
        pipeline.set_source(files[0])
        pipeline.set_surface(surface)
        pipeline.fit_mesh()
        pipeline.set_interpolation_kernel(sharpness=16)
        geometric = pipeline.sample_geometry(param_coords)
        dynamic.append(pipeline.sample_source(param_coords))
        exit()
        pattern = r"^tmp\.\d+\.scastonguay_(\d+\.\d+)$"

        t = []
        files = files[slice(args.t_start if hasattr(args, 'F') else None,
                            args.t_end if hasattr(args, 't_end') else None)]
        for file in files:
            t.append(float(re.match(pattern, file.stem).group(1)))
            pipeline.set_source(file)
            dynamic.append(pipeline.sample_source(param_coords))
        geometric['t'] = np.array(t)
        geometric.update(concatenate_dicts(dynamic))
        np.savez(output_path.joinpath(state.stem), **geometric)
        shutil.rmtree(source_path.joinpath(sample_name))


if __name__ == '__main__':
    main()
