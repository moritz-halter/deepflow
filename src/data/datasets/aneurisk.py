from pathlib import Path
from typing import Union, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.datasets.pt_dataset import PTDataset


def create_datasets(data_dir: Union[Path, str],
                    resolutions: Union[List[int], int]):
    if isinstance(resolutions, int):
        resolutions = [resolutions]
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    assert data_dir.exists(), f'{data_dir} is not a directory'

    for res in resolutions:
        print(res)
        ds_dir = data_dir.joinpath(f'aneurisk_{res}')
        if not ds_dir.exists():
            ds_dir.mkdir()
        ds_file = ds_dir.joinpath(f'dataset.pt')
        samples = data_dir.joinpath('samples')
        assert samples.exists(), f'{samples.as_posix()} is not a directory'
        u = []
        p = []
        chi = []
        mc = []
        eig = []
        sdf = []
        coords = []
        noise = []
        slices = None
        for sample in samples.iterdir():
            print(sample.stem)
            sample_dict = np.load(sample.as_posix())
            if slices is None:
                dims = sample_dict['p'].shape[:3]
                rate = [d // res for d in dims]
                slices = (slice(None, None, rate[0]), slice(None, None, rate[1]), slice(None, None, rate[2]), slice(None))
            u_tmp = sample_dict['u'][*slices]
            u.append(u_tmp)
            p.append(sample_dict['p'][*slices, None])
            chi.append(sample_dict['mask'][None, *slices[:3], None])
            mc.append(sample_dict['mean_curvature'][None, *slices[:3], None])
            eig.append(sample_dict['eigenvectors'][:32, *slices[:3], None])
            sdf.append(sample_dict['distance'][None, *slices[:3], None])
            coords.append(sample_dict['coords'][*slices[:3], None])
            n = np.random.randn(*u[-1].shape)
            Es = np.sum(np.square(u[-1]))
            En = np.sum(np.square(n))
            alpha = np.sqrt(Es/(10*En))
            noise.append(n * alpha)
        u = torch.tensor(np.stack(u, axis=0)).permute(0, 5, 1, 2, 3, 4)
        p = torch.tensor(np.stack(p, axis=0)).permute(0, 5, 1, 2, 3, 4)
        chi = torch.tensor(np.stack(chi, axis=0))
        mc = torch.tensor(np.stack(mc, axis=0))
        eig = torch.tensor(np.stack(eig, axis=0)).permute(0, 5, 1, 2, 3, 4)
        sdf = torch.tensor(np.stack(sdf, axis=0))
        coords = torch.tensor(np.stack(coords, axis=0))
        noise = torch.tensor(np.stack(noise, axis=0)).permute(0, 5, 1, 2, 3, 4)
        data_dict = {'u': u,
                     'p': p,
                     'chi': chi,
                     'mc': mc,
                     'eig': eig.type(torch.float32),
                     'sdf': sdf,
                     'coords': coords.type(torch.float32),
                     'noise': noise.type(torch.float32)}
        torch.save(data_dict, ds_file.as_posix())

class AneuriskDataset(PTDataset):
    def __init__(self,
                 root_dir: Union[Path, str],
                 train_test_split: float,
                 batch_size: int,
                 test_batch_size: List[int],
                 train_resolution: int,
                 test_resolutions: List[int],
                 geometric_prior_mode: int = 0,
                 geometric_prior: str = 'mean_curvature',
                 super_resolution_rate=None,
                 super_resolution_dim='space',
                 domain_flag=False,
                 n_t: int = 0,
                 noise: bool = False):
        super().__init__(root_dir=root_dir,
                         train_test_split=train_test_split,
                         dataset_name='aneurisk',
                         batch_size=batch_size,
                         test_batch_sizes=test_batch_size,
                         train_resolution=train_resolution,
                         test_resolutions=test_resolutions,
                         n_t=n_t,
                         geometric_prior_mode=geometric_prior_mode,
                         geometric_prior=geometric_prior,
                         super_resolution_rate=super_resolution_rate,
                         super_resolution_dim=super_resolution_dim,
                         domain_flag=domain_flag,
                         noise=noise)

def load_aneurisk_dataset(train_test_split: float,
                          batch_size: int,
                          test_batch_sizes: List[int],
                          data_root: Union[str, Path],
                          train_resolution: int = 64,
                          test_resolutions: List[int] = [64],
                          n_t: int = 0,
                          geometric_prior_mode: int = 0,
                          geometric_prior: str = 'mean_curvature',
                          super_resolution_rate: int = None,
                          super_resolution_dim: str = 'space',
                          domain_flag: bool = False,
                          noise: bool = False):

    dataset = AneuriskDataset(root_dir=data_root,
                              train_test_split=train_test_split,
                              batch_size=batch_size,
                              test_batch_size=test_batch_sizes,
                              train_resolution=train_resolution,
                              test_resolutions=test_resolutions,
                              n_t=n_t,
                              geometric_prior_mode=geometric_prior_mode,
                              geometric_prior=geometric_prior,
                              super_resolution_rate=super_resolution_rate,
                              super_resolution_dim=super_resolution_dim,
                              domain_flag=domain_flag,
                              noise=noise)

    train_loader = DataLoader(dataset.train_db,
                              batch_size=batch_size,
                              num_workers=0,
                              pin_memory=True,
                              persistent_workers=False)

    test_loaders = {}
    for res, test_bsize in zip(test_resolutions, test_batch_sizes):
        test_loaders[res] = DataLoader(dataset.test_dbs[res],
                                       batch_size=test_bsize,
                                       shuffle=False,
                                       num_workers=0,
                                       pin_memory=True,
                                       persistent_workers=False)

    return train_loader, test_loaders, dataset.data_processor, dataset.t_out


