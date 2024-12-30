import os
from enum import Enum
from pathlib import Path

import gmsh
import numpy as np
import networkx as nx


def spherical_to_cartesian(spherical: np.ndarray) -> np.ndarray:
    return np.stack([spherical[..., 0] * np.sin(spherical[..., 1]) * np.cos(spherical[..., 2]),
                     spherical[..., 0] * np.sin(spherical[..., 1]) * np.sin(spherical[..., 2]),
                     spherical[..., 0] * np.cos(spherical[..., 1])], axis=-1)


def cylindrical_to_cartesian(cylindrical: np.ndarray) -> np.ndarray:
    return np.stack([cylindrical[..., 0] * np.cos(cylindrical[..., 1]),
                     cylindrical[..., 0] * np.sin(cylindrical[..., 1]),
                     cylindrical[..., 2]], axis=-1)


def rotate(points: np.ndarray, axis: np.ndarray, theta: float) -> np.ndarray:
    return (points * np.cos(theta) + np.cross(axis, points)
            * np.sin(theta) + np.einsum('ij,i -> ij',
                                        np.tile(np.reshape(axis, (1, 3)), (points.shape[0], 1)),
                                        np.dot(points, axis) * (1 - np.cos(theta))))


def get_transformation_matrix(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    if np.all(np.isclose(u, v)):
        return np.identity(3)
    elif np.all(np.isclose(u, -v)):
        return -np.identity(3)
    angle = np.acos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))
    if angle == 0:
        return np.identity(3) * np.linalg.norm(v) / np.linalg.norm(u)
    normal = np.cross(u, v)
    K = np.array([
        [0, -normal[2], normal[1]],
        [normal[2], 0, -normal[0]],
        [-normal[1], normal[0], 0]
    ])
    normal /= np.linalg.norm(normal)
    return (np.identity(3) + np.sin(angle) * K
            + (1 - np.cos(angle)) * np.linalg.matrix_power(K, 2)) * np.linalg.norm(v) / np.linalg.norm(u)


class MeshGenerator:
    class Type(Enum):
        BOX = 'BOX'
        SPHERE = 'SPHERE'
        CYLINDER = 'CYLINDER'
        DECOMPOSED_SPHERE = 'DECOMPOSED_SPHERE'
        DECOMPOSED_CYLINDER = 'DECOMPOSED_CYLINDER'

    @staticmethod
    def __initialize() -> None:
        if not gmsh.is_initialized():
            gmsh.initialize()

    @staticmethod
    def __create_box_geometry():
        gmsh.model.occ.add_box(x=-0.5, y=-0.5, z=-0.5, dx=1, dy=1, dz=1)
        gmsh.model.occ.synchronize()

    @staticmethod
    def __create_sphere_geometry():
        center = gmsh.model.occ.add_point(0, 0, 0)
        t = np.atan(np.sqrt(2))
        corners = [gmsh.model.occ.add_point(*v) for v in spherical_to_cartesian(np.array([
            [1, np.pi - t, 3 * np.pi / 4],
            [1, np.pi - t, 5 * np.pi / 4],
            [1, np.pi - t, 7 * np.pi / 4],
            [1, np.pi - t, 1 * np.pi / 4],
            [1, t, 3 * np.pi / 4],
            [1, t, 5 * np.pi / 4],
            [1, t, 7 * np.pi / 4],
            [1, t, 1 * np.pi / 4],
        ]))]
        curves = [gmsh.model.occ.add_circle_arc(corners[line[0]],
                                                center,
                                                corners[line[1]]) for line in [(0, 1), (1, 2), (2, 3), (3, 0),
                                                                               (4, 5), (5, 6), (6, 7), (7, 4),
                                                                               (0, 4), (1, 5), (2, 6), (3, 7)]]
        support_coords = spherical_to_cartesian(np.array([
            [1, 0, 0],
            [1, 5 / 32 * np.pi, 0],
            [1, 5 / 32 * np.pi, 2 * np.pi / 8],
            [1, 5 / 32 * np.pi, 4 * np.pi / 8],
            [1, 5 / 32 * np.pi, 6 * np.pi / 8],
            [1, 5 / 32 * np.pi, 8 * np.pi / 8],
            [1, 5 / 32 * np.pi, 10 * np.pi / 8],
            [1, 5 / 32 * np.pi, 12 * np.pi / 8],
            [1, 5 / 32 * np.pi, 14 * np.pi / 8]
        ]))
        rotations = [
            (np.array([1, 0, 0]), np.pi),
            (np.array([0, 1, 0]), 3 * np.pi / 2),
            (np.array([1, 0, 0]), np.pi / 2),
            (np.array([0, 1, 0]), np.pi / 2),
            (np.array([1, 0, 0]), 3 * np.pi / 2),
            (np.array([1, 0, 0]), 0)
        ]
        support_tags = [gmsh.model.occ.add_point(*v) for v in np.concatenate(
            [rotate(support_coords, axis, theta) for axis, theta in rotations], axis=0
        )]
        wire_tags = [gmsh.model.occ.add_curve_loop(l) for l in [
            [curves[0], curves[1], curves[2], curves[3]],
            [curves[0], curves[9], curves[4], curves[8]],
            [curves[1], curves[10], curves[5], curves[9]],
            [curves[2], curves[11], curves[6], curves[10]],
            [curves[3], curves[8], curves[7], curves[11]],
            [curves[4], curves[5], curves[6], curves[7]]
        ]]
        surfaces = [gmsh.model.occ.add_surface_filling(w, pointTags=support_tags[9 * i:9 * (i + 1)])
                    for i, w in enumerate(wire_tags)]
        gmsh.model.occ.add_volume([gmsh.model.occ.add_surface_loop(surfaces)])
        gmsh.model.occ.remove([(0, t) for t in support_tags])
        gmsh.model.occ.remove([(0, center)])
        gmsh.model.occ.synchronize()

    @staticmethod
    def __create_cylinder_geometry():
        center_a = gmsh.model.occ.add_point(0, 0, -0.5)
        center_b = gmsh.model.occ.add_point(0, 0, 0.5)
        corners = [gmsh.model.occ.add_point(*v) for v in cylindrical_to_cartesian(np.array([
            [1, 3 * np.pi / 4, -0.5],
            [1, 5 * np.pi / 4, -0.5],
            [1, 7 * np.pi / 4, -0.5],
            [1, 1 * np.pi / 4, -0.5],
            [1, 3 * np.pi / 4, 0.5],
            [1, 5 * np.pi / 4, 0.5],
            [1, 7 * np.pi / 4, 0.5],
            [1, 1 * np.pi / 4, 0.5]
        ]))]
        curves = [gmsh.model.occ.add_circle_arc(a, c, b) for a, c, b in [
            (corners[0], center_a, corners[1]),
            (corners[1], center_a, corners[2]),
            (corners[2], center_a, corners[3]),
            (corners[3], center_a, corners[0]),
            (corners[4], center_b, corners[5]),
            (corners[5], center_b, corners[6]),
            (corners[6], center_b, corners[7]),
            (corners[7], center_b, corners[4])
        ]] + [gmsh.model.occ.add_line(a, b) for a, b in [
            (corners[0], corners[4]),
            (corners[1], corners[5]),
            (corners[2], corners[6]),
            (corners[3], corners[7])
        ]]

        support_coords = cylindrical_to_cartesian(np.array([
            [1, np.pi / 4 + np.pi / 6, -0.25],
            [1, np.pi / 4 + 2 * np.pi / 6, -0.25],
            [1, np.pi / 4 + np.pi / 6, 0.25],
            [1, np.pi / 4 + 2 * np.pi / 6, 0.25]
        ]))
        rotations = [
            (np.array([0, 0, 1]), np.pi / 2),
            (np.array([0, 0, 1]), np.pi),
            (np.array([0, 0, 1]), 3 * np.pi / 2),
            (np.array([0, 0, 1]), 2 * np.pi)
        ]
        support_tags = [gmsh.model.occ.add_point(*v) for v in np.concatenate(
            [rotate(support_coords, axis, theta) for axis, theta in rotations], axis=0
        )]
        wire_tags = [gmsh.model.occ.add_curve_loop(l) for l in [
            [curves[0], curves[9], curves[4], curves[8]],
            [curves[1], curves[10], curves[5], curves[9]],
            [curves[2], curves[11], curves[6], curves[10]],
            [curves[3], curves[8], curves[7], curves[11]],
            [curves[0], curves[1], curves[2], curves[3]],
            [curves[4], curves[5], curves[6], curves[7]]
        ]]
        surfaces = ([gmsh.model.occ.add_surface_filling(w, pointTags=support_tags[4 * i:4 * (i + 1)])
                     for i, w in enumerate(wire_tags[:4])]
                    + [gmsh.model.occ.add_plane_surface([w]) for w in wire_tags[4:]])
        gmsh.model.occ.add_volume([gmsh.model.occ.add_surface_loop(surfaces)])
        gmsh.model.occ.remove([(0, t) for t in support_tags])
        gmsh.model.occ.remove([(0, center_a)])
        gmsh.model.occ.remove([(0, center_b)])
        gmsh.model.occ.synchronize()

    @staticmethod
    def __create_decomp_sphere_geometry():
        pass

    @staticmethod
    def __create_decomp_cylinder_geometry():
        pass

    @staticmethod
    def save(path: Path, model_name: str = None, extension: str = 'vtk') -> None:
        if model_name is not None:
            gmsh.model.set_current(model_name)
        gmsh.write(os.path.join(path, f'mesh.{extension}'))
        volume_dim_tags = gmsh.model.get_entities(3)
        for i, volume_dim_tag in enumerate(volume_dim_tags):
            t, _, _ = gmsh.model.mesh.get_nodes(*volume_dim_tag, returnParametricCoord=False)
            N = np.round(np.cbrt(len(t)), 0).astype(int).item() + 2
            n_surface = np.pow((N - 2), 2)
            n_volume = np.pow((N - 2), 3)
            grid = np.zeros((N, N, N), dtype=int)
            graph = nx.Graph()
            _, surface_tags = gmsh.model.get_adjacencies(*volume_dim_tag)
            for surface_tag in surface_tags:
                _, curve_tags = gmsh.model.get_adjacencies(2, surface_tag)
                for curve_tag in curve_tags:
                    _, point_tags = gmsh.model.get_adjacencies(1, curve_tag)
                    for point_tag in point_tags:
                        node_tags, coords, _ = gmsh.model.mesh.get_nodes(0, point_tag)
                        if node_tags[0] not in graph:
                            parametric_coords = (
                                0 if coords[0] < 0.0 else N - 1,
                                0 if coords[1] < 0.0 else N - 1,
                                0 if coords[2] < 0.0 else N - 1
                            )
                            grid[parametric_coords] = node_tags[0]
                            graph.add_node(node_tags[0].item(), coords=coords, param_coords=parametric_coords)
                    node_tags, coords, parametric_coords = gmsh.model.mesh.get_nodes(1, curve_tag,
                                                                                     includeBoundary=True)
                    coords = np.reshape(coords, (-1, 3))
                    corner_pc = np.array([np.where(grid == n) for n in node_tags[-2:]]).reshape((2, 3))
                    axis = np.where(corner_pc[0, ...] != corner_pc[1, ...])[0].item()
                    if axis == 0:
                        if coords[0, 0] > coords[-3, 0]:
                            node_tags = np.flip(node_tags[:(N - 2)])
                        grid[1:-1, corner_pc[0, 1], corner_pc[0, 2]] = node_tags[:(N - 2)]
                    elif axis == 1:
                        if coords[0, 1] > coords[-3, 1]:
                            node_tags = np.flip(node_tags[:(N - 2)])
                        grid[corner_pc[0, 0], 1:-1, corner_pc[0, 2]] = node_tags[:(N - 2)]
                    else:
                        if coords[0, 2] > coords[-3, 2]:
                            node_tags = np.flip(node_tags[:(N - 2)])
                        grid[corner_pc[0, 0], corner_pc[0, 1], 1:-1] = node_tags[:(N - 2)]
                    # parametric_coords = [tuple(i.item() for i in np.where(grid == t)) for t in node_tags[:(N - 2)]]
                    # for t, c, pc in zip(node_tags[:(N - 2)], coords[:(N - 2), ...], parametric_coords):
                    #     graph.add_node(t.item(), coords=c, param_coords=pc)
                node_tags, coords, parametric_coords = gmsh.model.mesh.get_nodes(2, surface_tag,
                                                                                 includeBoundary=True)
                coords = np.reshape(coords, (-1, 3))
                corner_pc = np.array([np.where(grid == n) for n in node_tags[-3:]]).reshape((3, 3))
                fixed_axis = np.where(np.logical_and(corner_pc[0, ...] == corner_pc[1, ...],
                                                     corner_pc[0, ...] == corner_pc[2, ...]))[0].item()
                node_tags_grid = np.reshape(node_tags[:n_surface], (N - 2,) * 2)
                coords_grid = np.reshape(coords[:n_surface, ...], (N - 2,) * 2 + (3,))
                if fixed_axis == 0:
                    if np.isclose(coords_grid[0, 0, 1], coords_grid[-1, 0, 1], rtol=1e-1):
                        node_tags_grid = np.swapaxes(node_tags_grid, 0, 1)
                        coords_grid = np.swapaxes(coords_grid, 0, 1)
                    if coords_grid[0, 0, 1] > coords_grid[-1, -1, 1]:
                        node_tags_grid = np.flip(node_tags_grid, axis=0)
                    if coords_grid[0, 0, 2] > coords_grid[-1, -1, 2]:
                        node_tags_grid = np.flip(node_tags_grid, axis=1)
                    grid[corner_pc[0, 0], 1:-1, 1:-1] = node_tags_grid
                elif fixed_axis == 1:
                    if np.isclose(coords_grid[0, 0, 0], coords_grid[-1, 0, 0], rtol=1e-1):
                        node_tags_grid = np.swapaxes(node_tags_grid, 0, 1)
                        coords_grid = np.swapaxes(coords_grid, 0, 1)
                    if coords_grid[0, 0, 0] > coords_grid[-1, -1, 0]:
                        node_tags_grid = np.flip(node_tags_grid, axis=0)
                    if coords_grid[0, 0, 2] > coords_grid[-1, -1, 2]:
                        node_tags_grid = np.flip(node_tags_grid, axis=1)
                    grid[1:-1, corner_pc[0, 1], 1:-1] = node_tags_grid
                else:
                    if np.isclose(coords_grid[0, 0, 0], coords_grid[-1, 0, 0], rtol=1e-1):
                        node_tags_grid = np.swapaxes(node_tags_grid, 0, 1)
                        coords_grid = np.swapaxes(coords_grid, 0, 1)
                    if coords_grid[0, 0, 0] > coords_grid[-1, -1, 0]:
                        node_tags_grid = np.flip(node_tags_grid, axis=0)
                    if coords_grid[0, 0, 1] > coords_grid[-1, -1, 1]:
                        node_tags_grid = np.flip(node_tags_grid, axis=1)
                    grid[1:-1, 1:-1, corner_pc[0, 2]] = node_tags_grid
                # parametric_coords = [tuple(i.item() for i in np.where(grid == t)) for t in node_tags[:n_surface]]
                # for t, c, pc in zip(node_tags[:n_surface], coords[:n_surface, ...], parametric_coords):
                #     graph.add_node(t.item(), coords=c, param_coords=pc)
            node_tags, coords, _ = gmsh.model.mesh.get_nodes(*volume_dim_tag, includeBoundary=True)
            coords = np.reshape(coords, (-1, 3))
            coords_grid = np.reshape(coords[:n_volume, ...], (N - 2,) * 3 + (3,))
            node_tags_grid = np.reshape(node_tags[:n_volume], (N - 2,) * 3)
            axis_a = np.where(np.isclose(coords_grid[0, 0, 0, ...], -coords_grid[-1, 0, 0, ...], rtol=1e-1))[0].item()
            axis_b = np.where(np.isclose(coords_grid[0, 0, 0, ...], -coords_grid[0, -1, 0, ...], rtol=1e-1))[0].item()
            if axis_a == 1:
                node_tags_grid = np.swapaxes(node_tags_grid, 0, 1)
                coords_grid = np.swapaxes(coords_grid, 0, 1)
                if axis_b == 2:
                    node_tags_grid = np.swapaxes(node_tags_grid, 0, 2)
                    coords_grid = np.swapaxes(coords_grid, 0, 2)
            elif axis_a == 2:
                node_tags_grid = np.swapaxes(node_tags_grid, 0, 2)
                coords_grid = np.swapaxes(coords_grid, 0, 2)
                if axis_b == 0:
                    node_tags_grid = np.swapaxes(node_tags_grid, 0, 1)
                    coords_grid = np.swapaxes(coords_grid, 0, 1)
            else:
                if axis_b == 2:
                    node_tags_grid = np.swapaxes(node_tags_grid, 1, 2)
                    coords_grid = np.swapaxes(coords_grid, 1, 2)
            if coords_grid[0, 0, 0, 0] > coords_grid[-1, -1, -1, 0]:
                node_tags_grid = np.flip(node_tags_grid, axis=0)
            if coords_grid[0, 0, 0, 1] > coords_grid[-1, -1, -1, 1]:
                node_tags_grid = np.flip(node_tags_grid, axis=1)
            if coords_grid[0, 0, 0, 2] > coords_grid[-1, -1, -1, 2]:
                node_tags_grid = np.flip(node_tags_grid, axis=2)
            grid[1:-1, 1:-1, 1:-1] = node_tags_grid
            # parametric_coords = [tuple(i.item() for i in np.where(grid == t)) for t in node_tags[:n_volume]]
            # for t, c, pc in zip(node_tags[:n_volume], coords[:n_volume, ...], parametric_coords):
            #     graph.add_node(t.item(), coords=c, param_coords=pc)
            # types, _, elements = gmsh.model.mesh.get_elements(*volume_dim_tag)
            # elements = np.reshape(elements[np.where(types == 4)[0].item()], (-1, 4))
            # for node_tags in elements:
            #     edges = itertools.combinations(node_tags, 2)
            #     for edge in edges:
            #         if edge[0] not in graph[edge[1]]:
            #             coords_a, _, _, _ = gmsh.model.mesh.get_node(edge[0])
            #             coords_b, _, _, _ = gmsh.model.mesh.get_node(edge[1])
            #             graph.add_edge(edge[0].item(), edge[1].item(),
            #                            d=np.linalg.norm(coords_a - coords_b))
            # corner_nodes = grid[(0, -1), (0, -1), (0, -1)]
            param_coords = np.stack(np.unravel_index(np.argsort(grid, axis=None), grid.shape), axis=-1)
            # param_coords = nx.get_node_attributes(graph, 'param_coords')
            # param_coords = np.array(list(dict(sorted(param_coords.items())).values()))
            np.save(os.path.join(path, 'param_coords'), param_coords)
            gmsh.model.remove()

    @staticmethod
    def create(mesh_type: 'MeshGenerator.Type', name: str = None) -> str:
        MeshGenerator.__initialize()
        gmsh.model.add(name if name is not None else mesh_type.name)
        if mesh_type == MeshGenerator.Type.BOX:
            MeshGenerator.__create_box_geometry()
        elif mesh_type == MeshGenerator.Type.SPHERE:
            MeshGenerator.__create_sphere_geometry()
        elif mesh_type == MeshGenerator.Type.CYLINDER:
            MeshGenerator.__create_cylinder_geometry()
        elif mesh_type == MeshGenerator.Type.DECOMPOSED_SPHERE:
            MeshGenerator.__create_decomp_sphere_geometry()
        elif mesh_type == MeshGenerator.Type.DECOMPOSED_CYLINDER:
            MeshGenerator.__create_decomp_cylinder_geometry()
        else:
            raise NotImplementedError
        return mesh_type.name if name is None else name

    @staticmethod
    def generate_mesh(n: int, name: str = None) -> None:
        if name is not None:
            gmsh.model.set_current(name)
        volume_tags = []
        for volume_dim_tag in gmsh.model.get_entities():
            dim, tag = volume_dim_tag
            if dim == 1:
                gmsh.model.mesh.set_transfinite_curve(tag, n)
            elif dim == 2:
                gmsh.model.mesh.set_transfinite_surface(tag)
            elif dim == 3:
                gmsh.model.mesh.set_transfinite_volume(tag)
                volume_tags.append(tag)
        gmsh.model.mesh.generate()
