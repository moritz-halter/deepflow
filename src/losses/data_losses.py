"""
data_losses.py contains code to compute standard data objective 
functions for training Neural Operators. 

By default, losses expect arguments y_pred (model predictions) and y (ground y.)
"""

import math
from typing import List

import torch

# loss function with rel/abs Lp loss
class LpLoss(object):
    """
    LpLoss provides the L-p norm between two 
    discretized d-dimensional functions
    """
    def __init__(self, d=1, p=2, L=2*math.pi, reduce_dims=0, reductions='sum'):
        """

        Parameters
        ----------
        d : int, optional
            dimension of data on which to compute, by default 1
        p : int, optional
            order of L-norm, by default 2
            L-p norm: [\sum_{i=0}^n (x_i - y_i)**p] ** (1/p)
        L : float or list, optional
            quadrature weights per dim, by default 2*math.pi
            either single scalar for each dim, or one per dim
        reduce_dims : int, optional
            dimensions across which to reduce for loss, by default 0
        reductions : str, optional
            whether to reduce each dimension above 
            by summing ('sum') or averaging ('mean')
        """
        super().__init__()

        self.d = d
        self.p = p

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims
        
        if self.reduce_dims is not None:
            if isinstance(reductions, str):
                assert reductions == 'sum' or reductions == 'mean'
                self.reductions = [reductions]*len(self.reduce_dims)
            else:
                for j in range(len(reductions)):
                    assert reductions[j] == 'sum' or reductions[j] == 'mean'
                self.reductions = reductions

        if isinstance(L, float):
            self.L = [L]*self.d
        else:
            self.L = L
    
    @property
    def name(self):
        return f"L{self.p}_{self.d}Dloss"
    
    def uniform_h(self, x):
        """uniform_h creates default normalization constants
        if none already exist.

        Parameters
        ----------
        x : torch.Tensor
            input data

        Returns
        -------
        h : list
            list of normalization constants per-dim
        """
        h = [0.0]*self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j]/x.size(-j)
        
        return h

    def reduce_all(self, x):
        """
        reduce x across all dimensions in self.reduce_dims 
        according to self.reductions

        Params
        ------
        x: torch.Tensor
            inputs
        """
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == 'sum':
                x = torch.sum(x, dim=self.reduce_dims[j], keepdim=True)
            else:
                x = torch.mean(x, dim=self.reduce_dims[j], keepdim=True)
        
        return x

    def abs(self, x, y, h=None):
        """absolute Lp-norm

        Parameters
        ----------
        x : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        h : float or list, optional
            normalization constants for reduction
            either single scalar or one per dimension
        """
        #Assume uniform mesh
        if h is None:
            h = self.uniform_h(x)
        else:
            if isinstance(h, float):
                h = [h]*self.d
        
        const = math.prod(h)**(1.0/self.p)
        diff = const*torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), \
                                              p=self.p, dim=-1, keepdim=False)

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()
            
        return diff

    def rel(self, x, y):
        """
        rel: relative LpLoss
        computes ||x-y||/||y||

        Parameters
        ----------
        x : torch.Tensor
            inputs
        y : torch.Tensor
            targets
        """
        if self.d != 0:
            diff = torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d),
                              p=self.p, dim=-1, keepdim=False)
            ynorm = torch.norm(torch.flatten(y, start_dim=-self.d), p=self.p, dim=-1, keepdim=False)
        else:
            diff = torch.norm(x - y, p=self.p, dim=-1, keepdim=False)
            ynorm = torch.norm(y, p=self.p, dim=-1, keepdim=False)
            ynorm = torch.clamp(ynorm, 1e-6)

        diff = diff/ynorm

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()

        return diff

    def __call__(self, y_pred, y, **kwargs):
        return self.rel(y_pred, y)


class PINNLoss(object):
    def __init__(self):
        super().__init__()

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs):
        return torch.sum((y_pred - y) ** 2)


class MSELoss(object):
    """
    MSELoss computes absolute mean-squared error between two tensors.
    """
    def __init__(self, reductions='sum'):
        super().__init__()
        allowed_reductions = ['sum', 'mean']
        assert reductions in allowed_reductions, f"error: expected `reductions` to be one of {allowed_reductions}, got {reductions}"
        self.reductions = reductions

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor, dim: List[int]=None, **kwargs):
        """MSE loss call 

        Parameters
        ----------
        y_pred : torch.Tensor
            tensor of predictions
        y : torch.Tensor
            ground truth, must be same shape as x
        dim : List[int], optional
            dimensions across which to compute MSE, by default None
        """
        if dim is None:
            dim = list(range(1, y_pred.ndim)) # no reduction across batch dim
        if self.reductions == 'sum':
            return torch.mean((y_pred - y) ** 2, dim=dim).sum() # sum of MSEs for each element
        elif self.reductions == 'mean':
            return torch.mean((y_pred - y) ** 2, dim=dim).mean()
