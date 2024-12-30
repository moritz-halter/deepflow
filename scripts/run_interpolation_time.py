from pathlib import Path

import numpy as np

from algebra.interpolation import interpolation_time


def run_interpolation(data, run_dir: Path, scale: int):
    # scale = 24
    if not run_dir.exists():
        run_dir.mkdir(parents=True)
    in_data = data['in_32']
    prediction = interpolation_time(in_data, scale)
    print(prediction.shape)
    print(scale)
    np.savez(run_dir.joinpath('pred.npz'),
             truth_32=data['truth_32'],
             coords_32=data['coords_32'],
             pred_32=prediction,
             in_32=data['in_32'])



def main():
    import time
    interpolation_modes = ['linear']
    runs_dir = Path('PATH/TO/TASK/DIRECTORY') # SET PATH HERE
    tasks = ['time_32']
    for task in tasks:
        default_dir = runs_dir.joinpath(f'{task}/dafno_edsr')
        task_dir = runs_dir.joinpath(task)
        for interpolation_mode in interpolation_modes:
            print(f'Interpolation mode: {interpolation_mode}')
            run_dir = task_dir.joinpath(interpolation_mode)
            start_time = time.time()
            run_interpolation(np.load(default_dir.joinpath('pred.npz')),
                              run_dir, int(task[-1]))
            print("--- %s seconds ---" % (time.time() - start_time))



if __name__ == '__main__':
    main()
