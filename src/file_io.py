import re
from pathlib import Path
from typing import Tuple

import xml.etree.ElementTree as eTree

import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonDataModel import vtkSphere
from vtkmodules.vtkFiltersCore import vtkPolyDataNormals
from vtkmodules.vtkIOLegacy import vtkPolyDataReader

from algebra.util import down_sample


def get_sample_data(name: str):
    samples_dir = Path('/home/moritz/Desktop/deepflow/data/samples')
    sample_file = samples_dir.joinpath(name) if name.endswith('.npz') else samples_dir.joinpath(f'{name}.npz')
    assert sample_file.is_file()
    data = np.load(sample_file)
    return data['coords'], data['eigenvectors'], data['u']


def get_surface_data(name: str, geometries_dir: Path):
    vtk_sample = geometries_dir.joinpath(f'{name}_surface.vtk')
    assert vtk_sample.is_file()
    vtk_reader = vtkPolyDataReader()
    vtk_reader.SetFileName(vtk_sample.as_posix())
    normal_filter = vtkPolyDataNormals()
    normal_filter.SetInputConnection(vtk_reader.GetOutputPort())
    normal_filter.Update()
    output = normal_filter.GetOutput()
    points = vtk_to_numpy(output.GetPoints().GetData())
    u = vtk_to_numpy(output.GetPointData().GetArray('U'))
    eigenvectors = vtk_to_numpy(output.GetPointData().GetArray('eigenvectors'))
    normals = vtk_to_numpy(output.GetPointData().GetArray('Normals'))
    wss = vtk_to_numpy(output.GetPointData().GetArray('wallShearStress'))
    mean_curvature = vtk_to_numpy(output.GetPointData().GetArray('Mean_Curvature'))
    return points[::1, :], normals[::1, :], eigenvectors[::1, :], wss[::1, :], u[::1, :], mean_curvature[::1]


def get_run_data(run_dir: Path, resolution: int):
    sample_filter = [0, 2, 3, 4, 5, 6, 7, 8]
    data = np.load(run_dir.joinpath('pred.npz'))
    if 'truth_32' in data.keys():
        prediction = data[f'pred_32'][sample_filter, ...]
        coords = data[f'coords_32'][sample_filter, ...]
        truth = data[f'truth_32']
    else:
        prediction = data[f'pred_{resolution}'][sample_filter, ...]
        coords = data[f'coords_{resolution}'][sample_filter, ...]
        truth = data[f'truth_{resolution}']
    if coords.shape[-1] == 3:
        coords = np.permute_dims(coords, (0, 5, 1, 2, 3, 4))
    truth = truth[sample_filter, ...]
    sampling_rate = prediction.shape[2] // resolution
    return (down_sample(coords, sampling_rate),
            down_sample(prediction, sampling_rate),
            down_sample(truth, sampling_rate))


def read_state_file(path: Path) -> Tuple[str, vtkSphere]:
    assert path.is_file(), '%s is not a file' % path.as_posix()
    root = eTree.parse(str(path)).getroot()

    sources = root.find("./ServerManagerState/ProxyCollection[@name='sources']")

    case_name = next(i.attrib['name'][:-6] for i in sources
                     if re.match(r'^C\d{4}\.vtk$', i.attrib['name']) is not None)

    clip_id = None
    case_name = None
    for item in sources:
        if re.match(r'^C\d{4}\.vtk$', item.attrib['name']) is not None:
            case_name = item.attrib['name']
        else:
            clip_id = int(item.attrib['id'])
    case_name = 'C0069/VTK/VTK/tmp.50711090.scastonguay_10.559999764.vtk'
    assert clip_id is not None and case_name is not None, 'Missing source item in %s' % path.as_posix()

    clip_function_id = int(root.find(f"./ServerManagerState/Proxy[@id='{clip_id}']"
                                     f"/Property[@name='ClipFunction']/Proxy").attrib['value'])
    clip_function = root.find(f"./ServerManagerState/Proxy[@id='{clip_function_id}']")

    center = np.array([float(clip_function.find("./Property[@name='Center']/Element[@index='0']").attrib['value']),
                       float(clip_function.find("./Property[@name='Center']/Element[@index='1']").attrib['value']),
                       float(clip_function.find("./Property[@name='Center']/Element[@index='2']").attrib['value'])])
    radius = float(clip_function.find("./Property[@name='Radius']/Element").attrib['value'])

    sphere = vtkSphere()
    sphere.SetCenter(center)
    sphere.SetRadius(radius)
    return case_name, sphere