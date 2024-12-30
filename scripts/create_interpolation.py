from pathlib import Path

import torch

from algebra.interpolation import interpolation_grid


def main():
    res = 32
    sampling_rates = [2, 4]
    set_name = 'aneurisk_' + str(res)
    base_dir = Path('PATH/TO/DATA/DIRECTORY').joinpath(set_name) # SET PATH HERE
    dataset = base_dir.joinpath('dataset.pt')
    data = torch.load(dataset.as_posix())
    u = data['u'].detach().numpy()
    for sampling_rate in sampling_rates:
        u_int = interpolation_grid(u[:, :, ::sampling_rate, ::sampling_rate, ::sampling_rate, :], u.shape[2], sampling_rate, 'rbf')
        data['u' + str(sampling_rate)] = torch.tensor(u_int, dtype=torch.float32)
        torch.save(data, base_dir.joinpath(f'dataset_rbf_{sampling_rate}_space.pt'))

if __name__ == '__main__':
    main()
