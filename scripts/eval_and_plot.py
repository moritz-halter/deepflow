import csv
from pathlib import Path
from typing import Union, List

import numpy as np

from algebra.gradient import gradient, divergence
from algebra.metrics import lp_diff, angular_error, wall_shear_stress
from algebra.util import get_grid_basis
from file_io import get_sample_data, get_run_data
from plots import eigenvectors_plot, flow_plot, create_flow_plot, create_p_v_t_plot, create_comparison_plot, \
    surface_mesh_plot, surface_plot, wall_shear_stress_plot, gather_samples_data, create_mse_plots

SAMPLE_IDX = 6
ELEV = -90
AZIM = 0
ROLL = 30


def process_run(points, predictions, truth, model: str, sample_names: List[str], save_path: Union[str, Path] = None, additional_plots: bool = False):

    surface_points, surface_normals, eigenvectors, surface_wss, surface_u = gather_samples_data(sample_names=sample_names, geometries_dir=geometries_dir)

    grid_basis, spacing = get_grid_basis(points, with_spacing=True)

    mse = lp_diff(predictions, truth)

    ae = angular_error(predictions, truth)
    if additional_plots:
        create_mse_plots(mse, save_path=save_path)
        grid_points, grid_eigenvectors, grid_u = get_sample_data('C0069')
        # on surface mesh points
        eigenvectors_plot(eigenvectors[SAMPLE_IDX],
                          surface_points[SAMPLE_IDX],
                          elev=-90,
                          azim=0,
                          roll=-30,
                          s=2000,
                          colormap='coolwarm',
                          wspace=-0.775,
                          save_path=save_path.joinpath('eigenvectors_surface.png'))

        grid_points_flat = grid_points.reshape(-1, grid_points.shape[-1])
        grid_eigenvectors_flat = grid_eigenvectors.reshape(-1, grid_eigenvectors.shape[-1])
        grid_eigenvectors_flat_norm = np.linalg.norm(grid_eigenvectors_flat, axis=-1)
        indices = np.where(grid_eigenvectors_flat_norm != 0)
        # on cartesian grid points
        eigenvectors_plot(grid_eigenvectors_flat[*indices, :],
                          grid_points_flat[*indices, :],
                          elev=-90,
                          azim=0,
                          roll=-30,
                          s=2000,
                          colormap='coolwarm',
                          wspace=-0.8,
                          save_path=save_path.joinpath('eigenvectors_grid.png'))
        flow_plot(points[SAMPLE_IDX], truth[SAMPLE_IDX],
                  save_path=save_path)

        surface_plot(surface_points[SAMPLE_IDX], surface_normals[SAMPLE_IDX], surface_wss[SAMPLE_IDX])

    grad_prediction = np.stack([gradient(predictions, spacing=spacing, dim=i) for i in range(3)], axis=1)
    grad_truth = np.stack([gradient(truth, spacing=spacing, dim=i) for i in range(3)], axis=1)

    div_pred = divergence(grad_prediction, grid_basis)
    div_truth = divergence(grad_truth, grid_basis)

    wss_prediction, indices = wall_shear_stress(grad_prediction, points, surface_normals, surface_points)
    wss_truth, _ = wall_shear_stress(grad_truth, points, surface_normals, surface_points)

    if additional_plots:
        wall_shear_stress_plot(wss_prediction[SAMPLE_IDX],
                               surface_points[SAMPLE_IDX][indices[SAMPLE_IDX]],
                               length=5e-4,
                               norm=False,
                               save_path=save_path)


    wss_diff = [wss_pred - wss_truth for wss_pred, wss_truth in zip(wss_prediction, wss_truth)]
    wss_diff_dist = [np.linalg.norm(diff, axis=-2) for diff in wss_diff]

    return {'model' : model,
            'mean_flow_diff_dist': np.mean(mse),
            'mean_angular_error': np.mean(ae),
            'mean_abs_div': np.mean(np.abs(div_pred)),
            'mean_abs_div_error': np.mean(np.abs(div_truth - div_pred)),
            'mean_wss_diff_dist': np.mean([np.mean(diff) for diff in wss_diff_dist])}


def main(samples: List[str], r_dir: Path, model_names: List[str], save_path: Union[Path, None] = None, refresh: bool = False):

    for task in model_names:
        print('Task:', task)
        eval_dir = r_dir.joinpath(task).joinpath('evaluation')
        if not eval_dir.exists():
            eval_dir.mkdir()
        csv_file = r_dir.joinpath(task).joinpath('evaluation/table.csv')
        if csv_file.is_file():
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                evaluations = [row for row in reader]
        else:
            evaluations = []
        task_dir = r_dir.joinpath(task)
        for run in task_dir.iterdir():
            if run.name != 'evaluation':
                print('Run:', run.name)
                additional_plots = False  # run.name == 'EXAMPLE_RUN_NAME'
                curr = process_run(*get_run_data(run, 32),
                                   model=run.name,
                                   sample_names=samples,
                                   additional_plots=additional_plots,
                                   save_path=save_path)
                match = next(iter(i for i, e in enumerate(evaluations) if curr['model'] == e['model']), None)
                if match is not None:
                    evaluations[match] = curr
                else:
                    evaluations.append(curr)

                if len(evaluations) > 0:
                    print(f'saving {len(evaluations)} evaluations')
                    csv_keys = list(evaluations[0].keys())
                    with open(csv_file, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=csv_keys)
                        writer.writeheader()
                        for evaluation in evaluations:
                            writer.writerow(evaluation)


if __name__ == '__main__':
    tasks = ['space_noise_free_3'] # different training scenarios
    runs = ['idafno_edsr_eig_1', 'ifno_edsr', 'edsr', 'srcnn', 'linear'] # five methods to compare
    sample_names = ['C0045', 'C0048', 'C0020_1', 'C0056', 'C0043', 'C0079', 'C0069', 'C0080_1'] # samples to include in order (should be the evaluation samples in the right order)
    images_dir = Path('PATH/TO/IMAGE/DIRECTORY') # SET PATH HERE
    geometries_dir = Path('PATH/TO/GEOMETRY/DIRECTORY') # SET PATH HERE
    runs_dir = Path('PATH/TO/BASE/CHECKPOINT/DIRECTORY') # SET PATH HERE
    main(samples=sample_names, model_names=tasks, r_dir=runs_dir, save_path=images_dir)
    create_p_v_t_plot(samples_dir=Path('PATH/TO/SAMPLES/DIRECTORY'), # SET PATH HERE
                      save_dir=images_dir)
    create_flow_plot(ds_file=Path('PATH/TO/dataset.pt'), # SET PATH HERE
                     save_dir=images_dir)
    create_comparison_plot(task_dir=Path('PATH/TO/TASK/DIRECTORY'), # SET PATH HERE
                           save_dir=images_dir,
                           sample_names=sample_names,
                           runs=runs,
                           geometries_dir=geometries_dir)
    surface_mesh_plot(sample_name='C0069',
                      surface_dir=geometries_dir,
                      save_dir=images_dir)
