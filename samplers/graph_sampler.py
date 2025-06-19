import matplotlib.pyplot as plt
from torch import nn
from typing import List, Tuple, Any, Union
import torch
from utils.graph import DenseData, dense_collate_fn
from .base import SamplerWithBuffer
from utils.plot import plot_graph


class GraphSampler(SamplerWithBuffer):

    NODE_FEATURES_INTERVAL = None
    EDGE_FEATURES_INTERVAL = (1e-4, 1)

    def __init__(self, max_num_nodes=40, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_node_features = None
        self.num_edge_features = None
        self.max_num_nodes = max_num_nodes

    def _MCMC_generation(self, model: nn.Module, steps: int, step_size: float, labels: torch.Tensor,
                         starting_x: DenseData, is_training) -> DenseData:
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
        sample = starting_x
        sample.x.requires_grad_()

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

            sample.x.data.add_(- (step_size * sample.x.grad) + noise_x)
            sample.adj.data.add_(- (step_size * sample.adj.grad) + noise_adj)

            sample.x.grad.zero_()
            sample.adj.grad.zero_()

            with torch.no_grad():
                sample.x, sample.adj = self._normalize_graph(sample.x, sample.adj)

            sample.adj.requires_grad_()
            sample.x.requires_grad_()

        sample.detach_()

        return sample

    @torch.no_grad()
    def generate_random_batch(self, batch_size: int, device=None, collate: bool = True) -> (
            Union[List[Tuple[Any, torch.Tensor]], Tuple[Any, torch.Tensor]]):

        if device is None:
            device = self.device

        y = torch.randint(0, self.num_classes, size=(batch_size,), device=device)
        x = 2 * torch.randn((batch_size, self.max_num_nodes, self.num_node_features), device=device)
        adj = 0.1 + 0.1 * torch.randn((batch_size, self.max_num_nodes, self.max_num_nodes, self.num_edge_features),
                                      device=device)
        adj = adj.squeeze(-1)
        x, adj = self._normalize_graph(x, adj)
        mask = torch.ones((batch_size, self.max_num_nodes), dtype=torch.bool, device=device)

        if collate:
            return DenseData(x, adj, mask), y
        else:
            return [(DenseData(x[i], adj[i], mask[i]), y[i]) for i in range(batch_size)]

    @staticmethod
    def collate_fn(data_list: List[Tuple[Any, torch.Tensor]]) -> Tuple[Any, torch.Tensor]:
        return dense_collate_fn(data_list)

    def plot_sample(self, s: Tuple[DenseData, torch.Tensor], ax: plt.Axes) -> None:
        plot_graph(s[0], ax)
        ax.set_title(f'Label {s[1]}')

    def _normalize_graph(self, x, adj):
        # make symmetric
        new_adj = (adj + torch.transpose(adj, 1, 2))
        new_adj.div_(2)
        # remove self loops
        torch.diagonal(new_adj, dim1=1, dim2=2).zero_()
        # clamp between 0 and   1
        if self.EDGE_FEATURES_INTERVAL is not None:
            new_adj.clamp_(self.EDGE_FEATURES_INTERVAL[0], self.EDGE_FEATURES_INTERVAL[1])

        # clamp x features
        if self.NODE_FEATURES_INTERVAL is not None:
            new_x = torch.clamp(x, self.NODE_FEATURES_INTERVAL[0], self.NODE_FEATURES_INTERVAL[1])
        else:
            new_x = x

        return new_x, new_adj


class GraphSBMSampler(GraphSampler):

    def plot_sample(self, s: Tuple[DenseData, torch.Tensor], ax: plt.Axes) -> None:
        plot_graph(s[0], ax, n_communities=(s[1].item() + 1))
        ax.set_title(f'Label {s[1]}')