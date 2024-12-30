import itertools
import logging

from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkFiltersGeneral import vtkCurvatures
from vtkmodules.vtkIOLegacy import vtkPolyDataReader, vtkPolyDataWriter
from pathlib import Path
import numpy as np
import networkx as nx
from typing import Tuple, List, Set


def get_nodes(points: vtkPoints) -> Tuple[List[int], List[np.ndarray]]:
    idx_list = range(points.GetNumberOfPoints())
    coords = [points.GetPoint(idx) for idx in idx_list]
    return list(idx_list), coords


def get_edges(poly_data: vtkPolyData) -> Set[Tuple[int, int]]:
    res = set()
    for i in range(poly_data.GetNumberOfCells()):
        cell = poly_data.GetCell(i)
        cell_points = [cell.GetPointIds().GetId(i) for i in range(cell.GetPointIds().GetNumberOfIds())]
        if cell.GetCellType() == 5:
            edges = itertools.combinations(cell_points, 2)
            for edge in edges:
                res = res.union({(edge[0], edge[1])})
        elif cell.GetCellType() == 9:
            for j in range(len(cell_points)):
                if j < len(cell_points) - 1:
                    res = res.union({(cell_points[i], cell_points[j + 1])})
                else:
                    res = res.union({(cell_points[i], cell_points[0])})
        else:
            raise NotImplementedError(f"Cell type {cell.GetCellType()} is not implemented")
    return res


class GeometryPipeline:
    def __init__(self):
        self._reader = vtkPolyDataReader()
        self._curvature_filter = vtkCurvatures()
        self._curvature_filter.SetCurvatureTypeToMean()
        self._curvature_filter.SetInputConnection(self._reader.GetOutputPort())
        self._writer = vtkPolyDataWriter()

    def set_source(self, path: Path) -> None:
        assert path.is_file(), '%s is not a file' % path.as_posix()
        self._reader.SetFileName(path.as_posix())

    def set_writer_source(self, poly_data: vtkPolyData) -> None:
        self._writer.SetInputData(poly_data)

    def get_graph(self) -> nx.Graph:
        self._curvature_filter.Update()
        graph = nx.Graph()
        node_ids, coords = get_nodes(self._curvature_filter.GetOutput().GetPoints())
        graph.add_nodes_from(node_ids, coords=coords)
        graph.add_edges_from(get_edges(self._curvature_filter.GetOutput()))
        return graph

    def add_laplacian(self, poly_data: vtkPolyData) -> vtkPolyData:
        graph = self.get_graph()
        adj = nx.adjacency_matrix(graph)
        deg_plus = np.diag([np.sqrt(1 / d) if d != 0 else 0.0 for _, d in nx.degree(graph)])
        ident = np.diag(np.ones(adj.shape[0]))
        lap = ident - deg_plus @ adj @ deg_plus
        eig_vals, eig_vecs = np.linalg.eigh(lap)
        first_index = np.where(eig_vals > 1e-15)[0][0]
        logging.debug(eig_vals[:12])
        eig_vecs = numpy_to_vtk(eig_vecs[..., first_index:first_index + 64])
        eig_vecs.SetName('eigenvectors')
        poly_data.GetPointData().AddArray(eig_vecs)
        return poly_data

    def save_geometric_priors(self, path: Path) -> None:
        self._writer.SetFileName(path.as_posix())
        poly_data = self.add_laplacian(self._curvature_filter.GetOutput())
        self._writer.SetInputData(poly_data)
        self._writer.Update()
        self._writer.Write()

