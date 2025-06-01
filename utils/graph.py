from dataclasses import dataclass
from typing import Tuple, List, Union
import torch
import torch as th
from torch._C._nn import pad
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Dataset as pygDataset
from torch_geometric.utils import to_dense_adj


@dataclass
class DenseData:
    x: th.tensor
    adj: th.tensor
    mask: th.tensor

    def __repr__(self):
        return (
            f"DenseData("
            f"x={tuple(self.x.shape)}, "
            f"adj={tuple(self.adj.shape)}, "
            f"mask={tuple(self.mask.shape)})"
        )

    def __getitem__(self, index):
        return DenseData(
            self.x[index],
            self.adj[index],
            self.mask[index]
        )

    def __len__(self):
        return self.x.shape[0]

    def detach_(self):
        self.x = self.x.detach()
        self.adj = self.adj.detach()
        self.mask = self.mask.detach()

    def clone(self):
        return DenseData(self.x.clone(), self.adj.clone(), self.mask.clone())

    def cpu(self):
        return DenseData(self.x.cpu(), self.adj.cpu(), self.mask.cpu())


def dense_collate_fn(batch: List[Tuple[DenseData, th.Tensor]]) -> Tuple[DenseData, th.Tensor]:
    max_num_nodes = max([el[0].x.shape[0] for el in batch])
    x_list = []
    adj_list = []
    mask_list = []
    y_list = []

    for data, y in batch:
        n_nodes = data.x.shape[0]
        x = pad(data.x, (data.x.ndim - 1) * (0, 0) + (0, max_num_nodes - n_nodes)).unsqueeze(0)
        adj = pad(data.adj, (data.adj.ndim - 2) * (0, 0) + 2 * (0, max_num_nodes - n_nodes)).unsqueeze(0)
        adj.clamp_(0,1)
        mask = pad(data.mask, (0, max_num_nodes - n_nodes)).unsqueeze(0)

        x_list.append(x)
        adj_list.append(adj)
        mask_list.append(mask)
        y_list.append(y.unsqueeze(0))

    x_stacked = th.cat(x_list, dim=0)
    adj_stacked = th.cat(adj_list, dim=0)
    # TODO: dequantization with 2d gaussian convolution?
    adj_stacked.add_(0.1 * torch.randn_like(adj_stacked))
    mask_stacked = th.cat(mask_list, dim=0)
    y_stacked = th.cat(y_list, dim=0)

    return DenseData(x_stacked, adj_stacked, mask_stacked), y_stacked


class DenseGraphDataset(Dataset):

    """
    This class is wrapper of a pygDataset.
    """

    def __init__(self, pyg_dataset: Union[pygDataset, Subset[pygDataset]], get_dense_data_fun=None, get_y_fun=None):
        self._pyg_dataset = pyg_dataset

        if get_dense_data_fun is None:
            get_dense_data_fun = DenseGraphDataset._get_dense_graph

        if get_y_fun is None:
            get_y_fun = DenseGraphDataset._get_y

        self.data = []
        self.targets = []
        for el in self._pyg_dataset:
            self.data.append(get_dense_data_fun(el))
            self.targets.append(get_y_fun(el).squeeze(0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    @staticmethod
    def _get_y(data):
        return data.y

    @staticmethod
    def _get_dense_graph(data):
        x = data.x
        adj = to_dense_adj(
            data.edge_index,
            edge_attr=data.edge_attr if 'edge_attr' in data else None,
            max_num_nodes=x.shape[0]
        ).squeeze(0)
        mask = th.ones(
            x.shape[0],
            device=x.device,
            dtype=th.bool
        )

        return DenseData(x, adj.to(torch.float), mask)
