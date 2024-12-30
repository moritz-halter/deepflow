from abc import ABCMeta, abstractmethod

import torch

class DataProcessor(torch.nn.Module, metaclass=ABCMeta):
    def __init__(self):
        """DataProcessor exposes functionality for pre-
        and post-processing data during training or inference.

        To be a valid DataProcessor within the Trainer requires
        that the following methods are implemented:

        - to(device): load necessary information to device, in keeping
            with PyTorch convention
        - preprocess(data): processes data from a new batch before being
            put through a model's forward pass
        - postprocess(out): processes the outputs of a model's forward pass
            before loss and backward pass
        - wrap(self, model):
            wraps a model in preprocess and postprocess steps to create one forward pass
        - forward(self, x):
            forward pass providing that a model has been wrapped
        """
        super().__init__()

    @abstractmethod
    def to(self, device):
        pass

    @abstractmethod
    def preprocess(self, x):
        pass

    @abstractmethod
    def postprocess(self, x, data_dict):
        pass

    # default wrap method
    def wrap(self, model):
        self.model = model
        return self
    
    # default train and eval methods
    def train(self, val: bool = True):
        super().train(val)
        if self.model is not None:
            self.model.train()
    
    def eval(self):
        super().eval()
        if self.model is not None:
            self.model.eval()

    @abstractmethod
    def forward(self, x):
        pass


class DefaultDataProcessor(DataProcessor):
    """DefaultDataProcessor is a simple processor 
    to pre/post process data before training/inferencing a model.
    """
    def __init__(self, n_t=0):
        super().__init__()
        self.device = "cpu"
        self.model = None
        self.t = None
        self.n_t = n_t

    def to(self, device):
        self.device = device
        return self

    def preprocess(self, data_dict, batched=True):
        """preprocess a batch of data into the format
        expected in model's forward call

        Parameters
        ----------
        data_dict : dict
            input data dictionary with at least
            keys 'x' (inputs) and 'y' (ground truth)
        batched : bool, optional
            whether data contains a batch dim, by default True

        Returns
        -------
        dict
            preprocessed data_dict
        """
        for key in data_dict.keys():
            data_dict[key] = data_dict[key].to(self.device)
        if self.n_t == 0:
            _, c, h, w, d, self.t = data_dict['x'].size()
            data_dict['x'] = data_dict['x'].permute(0, 5, 1, 2, 3, 4).reshape(-1, c, h, w, d)
            if 'chi' in data_dict.keys():
                _, c, h, w, d, _ = data_dict['chi'].size()
                data_dict['chi'] = data_dict['chi'].repeat([1, 1, 1, 1, 1, self.t]).permute(0, 5, 1, 2, 3, 4).reshape(-1, c, h, w, d)
            if 'geo_feature' in data_dict.keys():
                _, c, h, w, d, _ = data_dict['geo_feature'].size()
                data_dict['geo_feature'] = data_dict['geo_feature'].repeat([1, 1, 1, 1, 1, self.t]).permute(0, 5, 1, 2, 3, 4).reshape(-1, c, h, w, d)
            return data_dict
        else:
            _, c, h, w, d, t = data_dict['x'].size()
            self.t = t // self.n_t
            data_dict['x'] = torch.stack(torch.split(data_dict['x'], [self.n_t] * self.t, dim=-1), dim=1).reshape(-1, c, h, w, d, self.n_t)
            t = data_dict['y'].size(-1) // self.n_t
            data_dict['y'] = data_dict['y'][:, :, :, :, :, :(t * self.n_t)]
            if 'chi' in data_dict.keys():
                _, c, h, w, d, _ = data_dict['chi'].size()
                data_dict['chi'] = torch.stack(
                    torch.split(data_dict['chi'].repeat(1, 1, 1, 1, 1, self.t), [1] * self.t, dim=-1),
                    dim=1).reshape(-1, c, h, w, d, 1)
            if 'geo_feature' in data_dict.keys():
                _, c, h, w, d, _ = data_dict['geo_feature'].size()
                data_dict['geo_feature'] = torch.stack(
                    torch.split(data_dict['geo_feature'].repeat(1, 1, 1, 1, 1, self.t), [1] * self.t, dim=-1),
                    dim=1).reshape(-1, c, h, w, d, 1)
            return data_dict

    def postprocess(self, output, data_dict):
        """postprocess model outputs and data_dict
        into format expected by training or val loss

        Parameters
        ----------
        output : torch.Tensor
            raw model outputs
        data_dict : dict
            dictionary containing single batch
            of data

        Returns
        -------
        out, data_dict
            postprocessed outputs and data dict
        """
        if len(list(output.size())) == 5:
            b, c, h, w, d = output.size()
            output = output.reshape(-1, self.t, c, h, w, d).permute(0, 2, 3, 4, 5, 1)
            return output, data_dict
        else:
            b, c, h, w, d, t = output.size()
            output = output.reshape(-1, self.t, c, h, w, d, t).permute(0, 2, 3, 4, 5, 1, 6).reshape(-1, c, h, w, d, t * self.t)
            return output, data_dict

    def forward(self, **data_dict):
        """forward call wraps a model
        to perform preprocessing, forward, and post-processing all in one call

        Returns
        -------
        output, data_dict
            postprocessed data for use in loss
        """
        data_dict = self.preprocess(data_dict)
        output = self.model(data_dict["x"])
        output, data_dict = self.postprocess(output, data_dict)
        return output, data_dict
