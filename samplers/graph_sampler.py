from torch import nn
import random
from typing import List, Tuple, Any, Union
import torch
import numpy as np
from lightning import LightningModule
from utils.graph import DenseData, dense_collate_fn
from .base import SamplerWithBuffer
import torchvision.transforms.functional as FT


class GraphSampler(SamplerWithBuffer):

    def __init__(self, min_num_nodes=10, max_num_nodes=40, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_node_features = None
        self.num_edge_features = None
        self.max_num_nodes = max_num_nodes
        self.min_num_nodes = min_num_nodes

    def _MCMC_generation(self, model: nn.Module, steps: int, step_size: float, labels: torch.Tensor, starting_x: Any) -> Any:
        """
        Function for generating new tensors via MCMC, given a model for :math:`E_{\\theta}`
        The MCMC algorith perform the following update:

        .. math::
            x_{k+1} = x_{k} - \\varepsilon \\nabla_{x} E_{\\theta} + \\omega

        :param model: Neural network to use for modeling :math:`E_{\theta}`
        :param starting_x: Batch of PyG Data to start from for sampling
        :param labels: Labels for conditional generation
        :param steps: Number of iterations in the MCMC algorithm.
        :param step_size: Learning rate :math:`\varepsilon`
        :return: Sampled Batch from the Energy distribution
        """
        sample: DenseData = starting_x
        sample.x.requires_grad_()

        # apply gaussian blurring?
        # sample.adj = FT.gaussian_blur(sample.adj, 3)
        sample.adj.requires_grad_()

        noise_x = torch.randn(sample.x.shape, device=labels.device)
        noise_adj = torch.randn(sample.adj.shape, device=labels.device)
        idx = torch.arange(labels.size(0), device=labels.device)

        # MCMC
        # batch.requires_grad = True
        for i in range(steps):
            noise_x.normal_(0, 0.005)
            noise_adj.normal_(0, 0.005)
            energy = -model(sample)[idx, labels]
            energy.sum().backward()
            #sample.x.grad.data.clamp_(-1, 1)
            #sample.adj.grad.data.clamp_(-1, 1)

            sample.x.data.add_(- (step_size * sample.x.grad) + noise_x)
            sample.adj.data.add_(- (step_size * sample.adj.grad) + noise_adj)

            sample.x.grad.zero_()
            sample.adj.grad.zero_()

            with torch.no_grad():
                sample.adj = (sample.adj + torch.transpose(sample.adj, 1, 2)) / 2
                sample.adj.clamp_(-10, 10)

            sample.adj.requires_grad_()

        sample.detach_()

        return sample

    @torch.no_grad()
    def generate_random_batch(self, batch_size: int, device=None, collate: bool = True) -> (
            Union[List[Tuple[Any, torch.Tensor]], Tuple[Any, torch.Tensor]]):

        if device is None:
            device = self.device

        num_nodes = torch.randint(10, self.max_num_nodes+1, size=(batch_size,)).tolist()
        data_list = []
        y = torch.randint(0, self.num_classes, size=(batch_size,), device=device)
        for i, n in enumerate(num_nodes):
            # TODO: is there a better way to generate random graphs? what about always using max_nodes?
            x = 2*torch.randn((n, self.num_node_features), device=device)
            A = 0.1*torch.randn((n, n), device=device)
            adj = (A.T + A) / 2
            mask = torch.ones((n,), dtype=torch.bool, device=device)
            data_list.append((DenseData(x, adj, mask), y[i]))

        if collate:
            return self.collate_fn(data_list)
        else:
            return data_list

    @staticmethod
    def collate_fn(data_list: List[Tuple[Any, torch.Tensor]]) -> Tuple[Any, torch.Tensor]:
        return dense_collate_fn(data_list)

    def plot_sample(self, s: DenseData) -> Any:
        if self.num_edge_features > 1:
            raise ValueError('Cannot plot an adjacency matrix with edge attributes that are not scalar')
        if s.adj.ndim == 2:
            # it is not a batch
            return s.adj.unsqueeze(0)
        if s.adj.ndim == 3:
            # it is a batch
            return s.adj.unsqueeze(1)