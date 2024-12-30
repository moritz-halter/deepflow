from pathlib import Path
from typing import Tuple, Dict
import itertools

import numpy as np
from vtkmodules.vtkCommonDataModel import vtkImplicitBoolean, vtkStaticPointLocator, vtkPointSet, vtkPolyData
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersCore import vtkPolyDataConnectivityFilter, vtkResampleWithDataSet, vtkImplicitPolyDataDistance
from vtkmodules.vtkFiltersGeneral import vtkTransformFilter, vtkCleanUnstructuredGrid, vtkClipDataSet, vtkOBBTree
from vtkmodules.vtkFiltersGeometry import vtkDataSetSurfaceFilter
from vtkmodules.vtkFiltersPoints import vtkPointInterpolator, vtkGaussianKernel
from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader, vtkPolyDataReader
from vtkmodules.util.numpy_support import vtk_to_numpy
from file_io import read_state_file
from algebra.util import get_transformation_matrix, flat_to_grid_dict


class DataPipeline:
    def __init__(self):
        self._reader = vtkUnstructuredGridReader()
        self._mesh_reader = vtkUnstructuredGridReader()
        self._surface_reader = vtkPolyDataReader()
        self._clip = vtkClipDataSet()
        self._clip.InsideOutOn()
        self._clip.SetValue(0)
        self._clip.SetInputConnection(self._reader.GetOutputPort())
        self._surface_filter = vtkDataSetSurfaceFilter()
        self._surface_filter.SetInputConnection(0, self._clip.GetOutputPort(0))
        self._connected_region_filter = vtkPolyDataConnectivityFilter()
        self._connected_region_filter.SetInputConnection(0, self._surface_filter.GetOutputPort(0))
        self._connected_region_filter.SetExtractionModeToLargestRegion()
        self._source_interpolator = vtkPointInterpolator()
        self._source_interpolator.SetSourceConnection(self._reader.GetOutputPort(0))
        self._resampler = vtkResampleWithDataSet()
        self._resampler.SetSourceConnection(self._reader.GetOutputPort())
        self._resampler.MarkBlankPointsAndCellsOff()
        self._point_cloud_sampler = vtkCleanUnstructuredGrid()
        self._point_cloud_sampler.SetInputConnection(0, self._clip.GetOutputPort(0))
        self._surface_interpolator = vtkPointInterpolator()
        self._surface_interpolator.SetSourceConnection(self._surface_reader.GetOutputPort(0))
        self._resampler.MarkBlankPointsAndCellsOn()
        self._distance_filter = vtkImplicitPolyDataDistance()
        self._transform_filter = vtkTransformFilter()
        self._transform_filter.SetInputConnection(self._mesh_reader.GetOutputPort(0))
        self._obb_tree = vtkOBBTree()
        self._obb_tree.SetMaxLevel(0)
        self._is_mesh = False

    def set_source(self, path: Path) -> None:
        assert path.is_file(), '%s is not a file' % path.as_posix()
        self._reader.SetFileName(path.as_posix())

    def set_state(self, path: Path) -> str:
        assert path.is_file(), '%s is not a file' % path.as_posix()
        name, sphere = read_state_file(path)
        boolean = vtkImplicitBoolean()
        boolean.AddFunction(sphere)
        self._clip.SetClipFunction(boolean)
        return name

    def set_mesh(self, path: Path) -> None:
        self._is_mesh = True
        assert path.is_file(), '%s is not a file' % path.as_posix()
        self._mesh_reader.SetFileName(path.as_posix())

    def set_surface(self, path: Path) -> None:
        assert path.is_file(), '%s is not a file' % path.as_posix()
        self._surface_reader.SetFileName(path.as_posix())

    def get_mesh_poly(self) -> vtkPointSet:
        self._transform_filter.Update()
        return self._transform_filter.GetOutput()

    def get_surface_poly(self) -> vtkPolyData:
        self._surface_reader.Update()
        return self._surface_reader.GetOutput()

    def get_source_surface_poly(self) -> vtkPolyData:
        self._connected_region_filter.Update(0)
        return self._connected_region_filter.GetOutput()

    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        surface_poly = self.get_source_surface_poly()
        self._obb_tree = vtkOBBTree()
        self._obb_tree.SetDataSet(surface_poly)
        self._obb_tree.BuildLocator()
        corner = [0.0, 0.0, 0.0]
        x = [0.0, 0.0, 0.0]
        y = [0.0, 0.0, 0.0]
        z = [0.0, 0.0, 0.0]
        size = [0.0, 0.0, 0.0]
        self._obb_tree.ComputeOBB(surface_poly, corner, x, y, z, size)
        return np.array(corner), np.array(x), np.array(y), np.array(z)

    def get_visual_bounding_box(self) -> vtkPolyData:
        self._obb_tree = vtkOBBTree()
        self._obb_tree.SetDataSet(self.get_source_surface_poly())
        self._obb_tree.BuildLocator()
        polydata = vtkPolyData()
        self._obb_tree.GenerateRepresentation(0, polydata)
        return polydata

    def fit_mesh(self) -> None:
        corner, x, y, z = self.get_bounding_box()
        transformation_matrix = get_transformation_matrix(corner, x, y, z)
        transform = vtkTransform()
        matrix = vtkMatrix4x4()
        matrix_2 = np.zeros((4, 4))
        for i, j in itertools.product(range(4), range(4)):
            matrix_2[i, j] = transform.GetMatrix().GetElement(i, j)
            matrix.SetElement(i, j, transformation_matrix[i, j].item())
        transform.SetMatrix(matrix)
        self._transform_filter.SetTransform(transform)
        self._transform_filter.Update()
        self._resampler.SetInputData(self._transform_filter.GetOutput())

    def set_interpolation_kernel(self, sharpness: float = 2, radius_factor: float = 0.25):
        output = self._resampler if self._is_mesh else self._point_cloud_sampler
        output.Update(0)
        output = output.GetOutput()

        n = output.GetNumberOfPoints()
        bounds = np.array(output.GetBounds()).reshape(3, 2)
        width = np.abs(bounds[:, 1] - bounds[:, 0])
        diag = np.linalg.norm(width)
        radius = diag / np.cbrt(n) * radius_factor

        locator = vtkStaticPointLocator()
        locator.SetDataSet(output)

        kernel = vtkGaussianKernel()
        kernel.SetSharpness(sharpness)
        kernel.SetRadius(radius)
        kernel.SetKernelFootprintToNClosest()

        self._surface_interpolator.SetLocator(locator)
        self._surface_interpolator.SetKernel(kernel)
        self._surface_interpolator.SetInputData(output)
        self._source_interpolator.SetLocator(locator)
        self._source_interpolator.SetKernel(kernel)
        self._source_interpolator.SetInputData(output)


    def get_distance(self) -> np.ndarray:
        points = self._transform_filter.GetOutput().GetPoints() \
            if self._is_mesh else self._point_cloud_sampler.GetOutput().GetPoints()
        self._distance_filter.SetInput(self.get_surface_poly())
        return -np.array([self._distance_filter.EvaluateFunction(
            points.GetPoint(i)
        ) for i in range(points.GetNumberOfPoints())])

    def sample_geometry(self, param_coords: np.ndarray = None) -> Dict[str, np.ndarray]:
        self._surface_interpolator.Update()
        point_data = self._surface_interpolator.GetOutput().GetPointData()
        distance = self.get_distance()
        if self._is_mesh:
            mask = vtk_to_numpy(self._resampler.GetOutput().GetPointData().GetArray('vtkValidPointMask'))
        else:
            mask = np.ones(distance.shape[:-1])
        res = {
            'mean_curvature': vtk_to_numpy(point_data.GetArray('Mean_Curvature')),
            'eigenvectors': vtk_to_numpy(point_data.GetArray('eigenvectors')),
            'coords': vtk_to_numpy(self._surface_interpolator.GetOutput().GetPoints().GetData()),
            'distance': distance,
            'mask': mask
        }
        return res if param_coords is None else flat_to_grid_dict(param_coords, res)

    def sample_source(self, param_coords: np.ndarray = None) -> Dict[str, np.ndarray]:
        if self._is_mesh:
            self._resampler.Update(0)
            data = self._resampler.GetOutput().GetPointData()
        else:
            self._point_cloud_sampler.Update()
            data = self._point_cloud_sampler.GetOutput().GetPointData()
        res = {
            'u': vtk_to_numpy(data.GetArray('U')),
            'p': vtk_to_numpy(data.GetArray('p'))
        }
        return res if param_coords is None else flat_to_grid_dict(param_coords, res)
