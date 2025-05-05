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

    def _generate_batch(self, model: nn.Module, labels: torch.Tensor, starting_x: Any, steps: int = 60,
                        step_size: float = 1.0) -> Any:
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
            sample.x.grad.data.clamp_(-0.1, 0.1)
            sample.adj.grad.data.clamp_(-1, 1)

            sample.x.data.add_(- (step_size * sample.x.grad) + noise_x)
            sample.adj.data.add_(- (step_size * sample.adj.grad) + noise_adj)

            sample.x.grad.zero_()
            sample.adj.grad.zero_()

            with torch.no_grad():
                sample.adj = (sample.adj + torch.transpose(sample.adj, 1, 2)) / 2
                sample.adj.data.clamp_(0, 1)
                sample.adj.data.round_()
            sample.adj.requires_grad_()

        sample.detach_()

        return sample

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
            A = torch.randint(0, 2, (n, n), device=device).to(torch.float)
            adj = A.T @ A
            mask = torch.ones((n,), dtype=torch.bool, device=device)
            data_list.append((DenseData(x, adj, mask), y[i]))

        if collate:
            return self.collate_fn(data_list)
        else:
            return data_list

    def collate_fn(self, data_list: List[Tuple[Any, torch.Tensor]]) -> Tuple[Any, torch.Tensor]:

        return dense_collate_fn(data_list)