import matplotlib.pyplot as plt
from torch import nn
from typing import List, Tuple, Any, Union
import torch
from utils.graph import DenseData, dense_collate_fn
from .graph_sampler import GraphSampler
from utils.plot import plot_graph
from utils.mol import make_valid
from utils.plot import plot_molecule


class MolSampler(GraphSampler):

    def __init__(self, max_num_nodes=9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: ???
        self.num_node_features = None
        self.num_edge_features = None
        self.max_num_nodes = max_num_nodes

    def _MCMC_generation(self, model: nn.Module, steps: int, step_size: float, labels: torch.Tensor,
                         starting_x: DenseData, is_training) -> DenseData:
        sample = super()._MCMC_generation(model, steps, step_size, labels, starting_x, is_training)
        if not is_training:
            sample = make_valid(sample)
        return sample

    def plot_sample(self, s: Tuple[DenseData, torch.Tensor], ax: plt.Axes) -> None:
        plot_molecule(s[0], ax=ax)
