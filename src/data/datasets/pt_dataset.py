from pathlib import Path
from typing import List, Union

import torch

from data.datasets.tensor_dataset import TensorDataset
from data.transforms.data_processors import DefaultDataProcessor


class PTDataset:
    """PTDataset is a base Dataset class for our library.
            PTDatasets contain input-output pairs a(x), u(x) and may also
            contain additional information, e.g. function parameters,
            input geometry or output query points.

            datasets may implement a download flag at init, which provides
            access to a number of premade datasets for sample problems provided
            in our Zenodo archive.

        All datasets are required to expose the following attributes after init:

        train_db: torch.utils.data.Dataset of training examples
        test_db:  ""                       of test examples
        data_processor: data.transforms.DataProcessor to process data examples
            optional, default is None
        """
    def __init__(self,
                 root_dir: Union[Path, str],
                 dataset_name: str,
                 train_test_split: float,
                 batch_size: int,
                 test_batch_sizes: List[int],
                 train_resolution: int,
                 test_resolutions: List[int] = [16, 32],
                 n_t: int = 0,
                 super_resolution_rate=None,
                 super_resolution_dim='space',
                 domain_flag: bool = False,
                 seed: int = 1234,
                 geometric_prior_mode: int = 0,
                 geometric_prior: str = 'eig',
                 noise: bool = False):
        """PTDataset.__init__

        Parameters
        ----------
        root_dir : Union[Path, str]
            root at which to download data files
        dataset_name : str
            prefix of pt data files to store/access
            number of test instances per test dataset
        train_test_split : float
            ratio between training and test data
        batch_size : int
            batch size of training set
        test_batch_sizes : List[int]
            batch size of test sets
        train_resolution : int
            resolution of data for training set
        test_resolutions : List[int], optional
            resolution of data for testing sets, by default [16,32]
        super_resolution_rate : int, optional
            sampling rate
        super_resolution_dim : str, by default 'space' in ['space', 'time']
        domain_flag : bool by default False
            in-/exclude the domain characteristic function in the dataset
        seed: int, by default 1234
            rng seed
        geometric_prior_mode : int, by default 0
            options for incorporating a geometric prior
        """

        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        self.root_dir = root_dir

        # save dataloader properties for later
        self.batch_size = batch_size
        self.test_resolutions = test_resolutions
        self.test_batch_sizes = test_batch_sizes
        assert geometric_prior in ['eig', 'mean_curvature', 'sdf']

        # Load train data
        data = torch.load(
            Path(root_dir).joinpath(f"{dataset_name}_{train_resolution}").joinpath('dataset.pt').as_posix(), weights_only=True
        )

        n_total = data['p'].size(0)
        n_train = round(n_total * train_test_split)
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(data['p'].size(0), generator=generator, dtype=torch.long)

        # optionally subsample along data indices
        ## Input subsampling
        # convert None and 0 to 1
        if not super_resolution_rate:
            super_resolution_rate = 1
        if super_resolution_dim == 'space' and not isinstance(super_resolution_rate, list):
            # expand subsampling rate along dims if one per dim is not provided
            super_resolution_rate = [super_resolution_rate] * 3
        # make sure there is one subsampling rate per data dim
        assert super_resolution_dim == 'time' or len(super_resolution_rate) == 3

        ## Output subsampling

        train_input_indices = [slice(None)] + [slice(None, None, rate) for rate in super_resolution_rate] + [slice(None, n_t)] \
            if super_resolution_dim == 'space' else [slice(None)] * 4 + [slice(None, n_t, super_resolution_rate)]
        x_train = (
            data['u'][[indices[:n_train]] + train_input_indices].type(torch.float32).clone()
            + data['noise'][[indices[:n_train]] + train_input_indices].type(torch.float32).clone()
        ) if noise else (
            data['u'][[indices[:n_train]] + train_input_indices].type(torch.float32).clone()
        )
        train_output_indices = [slice(None)] * 4 + [slice(None, n_t)]
        y_train = data['u'][[indices[:n_train]] + train_output_indices].clone()
        if domain_flag:
            chi_train = data['chi'][[indices[:n_train]] + train_output_indices].clone()
        else:
            chi_train = None
        geo_feature = None
        if geometric_prior_mode == 1:
            x_train = torch.cat([x_train, data[geometric_prior][[indices[:n_train]] + train_input_indices].type(torch.float32).clone().repeat(1, 1, 1, 1, 1, x_train.size(-1))], dim=1)
        elif geometric_prior_mode in [2, 3]:
            geo_feature = data[geometric_prior][[indices[:n_train]] + train_output_indices].type(torch.float32).clone()
        coords = data['coords'][[indices[:n_train]] + train_output_indices].clone()
        del data
        self.t_out = y_train.size(-1)
        # Save train dataset
        self._train_db = TensorDataset(
            x_train,
            y_train,
            coords=coords,
            chi=chi_train,
            geo_feature=geo_feature
        )
        if super_resolution_dim == 'space':
            self._data_processor = DefaultDataProcessor(n_t=n_t)
        else:
            self._data_processor = DefaultDataProcessor(n_t=n_t // super_resolution_rate)

        # load test data
        self._test_dbs = {}
        for res in test_resolutions:
            print(
                f"Loading test db for resolution {res} with {n_total - n_train} samples "
            )
            data = torch.load(Path(root_dir).joinpath(f"{dataset_name}_{res}").joinpath('dataset.pt').as_posix(), weights_only=True)

            # optionally subsample along data indices
            test_input_indices = [slice(None, None, None)] + [slice(None, None, rate) for rate in super_resolution_rate] + [slice(None, n_t)] \
                if super_resolution_dim == 'space' else [slice(None)] * 4 + [slice(None, n_t, super_resolution_rate)]
            x_test = (
                data["u"][[indices[n_train:]] + test_input_indices].type(torch.float32).clone()
                + data["noise"][[indices[n_train:]] + test_input_indices].type(torch.float32).clone()
            ) if noise else (
                data["u"][[indices[n_train:]] + test_input_indices].type(torch.float32).clone()
            )
            test_output_indices = [slice(None, None, None)] * 4 + [slice(None, n_t)]
            y_test = data["u"][[indices[n_train:]] + test_output_indices].clone()
            if domain_flag:
                chi = data["chi"][[indices[n_train:]] + test_output_indices].clone()
            else:
                chi = None
            mean_curvature = None
            if geometric_prior_mode == 1:
                x_test = torch.cat([x_test, data[geometric_prior][[indices[n_train:]] + test_input_indices].type(torch.float32).clone().repeat(1, 1, 1, 1, 1, x_test.size(-1))], dim=1)
            elif geometric_prior_mode in [2, 3]:
                mean_curvature = data[geometric_prior][[indices[n_train:]] + test_output_indices].type(torch.float32).clone()
            coords = data['coords'][[indices[n_train:]] + train_output_indices].clone()
            test_db = TensorDataset(
                x_test,
                y_test,
                coords=coords,
                chi=chi,
                geo_feature=mean_curvature
            )
            self._test_dbs[res] = test_db

    @property
    def data_processor(self):
        return self._data_processor

    @property
    def train_db(self):
        return self._train_db

    @property
    def test_dbs(self):
        return self._test_dbs

