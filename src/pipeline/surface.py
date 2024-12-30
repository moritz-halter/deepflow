from pathlib import Path

from vtkmodules.vtkCommonDataModel import vtkImplicitBoolean
from vtkmodules.vtkFiltersCore import vtkClipPolyData, vtkPolyDataConnectivityFilter
from vtkmodules.vtkFiltersGeometry import vtkDataSetSurfaceFilter
from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader, vtkPolyDataWriter

from file_io import read_state_file


class SurfacePipeline:
    def __init__(self):
        self._reader = vtkUnstructuredGridReader()
        self._surface_filter = vtkDataSetSurfaceFilter()
        self._surface_filter.SetInputConnection(0, self._reader.GetOutputPort(0))
        self._clip = vtkClipPolyData()
        self._clip.InsideOutOn()
        self._clip.SetValue(0)
        self._clip.SetInputConnection(self._surface_filter.GetOutputPort())
        self._connected_region_filter = vtkPolyDataConnectivityFilter()
        self._connected_region_filter.SetInputConnection(0, self._clip.GetOutputPort(0))
        self._connected_region_filter.SetExtractionModeToLargestRegion()
        self._writer = vtkPolyDataWriter()
        self._writer.SetInputConnection(self._connected_region_filter.GetOutputPort())

    def set_source(self, path: Path) -> None:
        assert path.is_file(), '%s is not a file' % path.as_posix()
        self._reader.SetFileName(path.as_posix())

    def set_state(self, path: Path) -> str:
        assert path.is_file(), '%s is not a file' % path.as_posix()
        name, sphere = read_state_file(path)
        sphere.SetRadius(sphere.GetRadius() * 1.75)
        boolean = vtkImplicitBoolean()
        boolean.AddFunction(sphere)
        self._clip.SetClipFunction(boolean)
        return name

    def write(self, path: Path) -> None:
        self._writer.SetFileName(path.as_posix())
        self._writer.Update()
        self._writer.Write()
