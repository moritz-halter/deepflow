from pathlib import Path

import numpy as np

from algebra.interpolation import interpolation_grid


def run_interpolation(data, interpolation_mode: str, run_dir: Path, scale: int):
    if not run_dir.exists():
        run_dir.mkdir(parents=True)
    truth = data[f'truth_32']
    in_data = data[f'in_32']
    in_data = in_data.reshape(truth.shape[0], in_data.shape[0] // truth.shape[0], truth.shape[1],
                              truth.shape[2] // scale, truth.shape[3] // scale, truth.shape[4] // scale,
                              in_data.shape[-1])
    in_data = np.permute_dims(in_data, (0, 2, 3, 4, 5, 6, 1)).reshape(in_data.shape[0], *in_data.shape[2:-1], -1)
    prediction = interpolation_grid(in_data, 32, scale, interpolation_mode)
    np.savez(run_dir.joinpath('pred.npz'),
             truth_32=data[f'truth_32'],
             coords_32=data[f'coords_32'],
             pred_32=prediction,
             in_32=data[f'in_32'])



def main():
    import time
    interpolation_modes = ['linear', 'rbf', 'dfi']
    runs_dir = Path('/home/moritz/Desktop/deepflow/runs') # SET PATH HERE
    tasks = ['space_noise_free_3']
    for task in tasks:
        default_dir = runs_dir.joinpath(f'{task}/idafno_edsr')
        task_dir = runs_dir.joinpath(task)
        for interpolation_mode in interpolation_modes:
            print(f'Interpolation mode: {interpolation_mode}')
            run_dir = task_dir.joinpath(interpolation_mode)
            start_time = time.time()
            run_interpolation(np.load(default_dir.joinpath('pred.npz')),
                              interpolation_mode, run_dir, int(task[-1]))
            print("--- %s seconds ---" % (time.time() - start_time))



if __name__ == '__main__':
    main()
