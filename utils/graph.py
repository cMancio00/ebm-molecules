from dataclasses import dataclass
from typing import Tuple, List
import torch
import torch as th
from torch._C._nn import pad
from torch.utils.data import Dataset
from torch_geometric.data import Dataset as pygDataset
from torch_geometric.utils import to_dense_adj

# TODO: consider store dense repr ons disk.
# TODO: we discard all the data attributes (except x, adj, node_mask) -> probably is enough for all datasets we use

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
        # TODO: dequantization with 2d gaussian convolution
        adj.add_(0.1*torch.randn_like(adj))
        adj.clamp_(0,1)
        mask = pad(data.mask, (0, max_num_nodes - n_nodes)).unsqueeze(0)

        x_list.append(x)
        adj_list.append(adj)
        mask_list.append(mask)
        y_list.append(y.unsqueeze(0))

    x_stacked = th.cat(x_list, dim=0)
    adj_stacked = th.cat(adj_list, dim=0)
    mask_stacked = th.cat(mask_list, dim=0)
    y_stacked = th.cat(y_list, dim=0)

    return DenseData(x_stacked, adj_stacked, mask_stacked), y_stacked


class DenseGraphDataset(Dataset):

    """
    This class is wrapper of a pygDataset.
    """

    def __init__(self, pyg_dataset: pygDataset):
        self._pyg_dataset = pyg_dataset

        self.data = []
        self.targets = []
        for el in self._pyg_dataset:
            el_dict = el.to_dict()
            x = el_dict.pop('x')
            adj = to_dense_adj(
                el_dict.pop('edge_index'),
                edge_attr=el_dict.pop('edge_attr', None),
                max_num_nodes=x.shape[0]
            ).squeeze(0)
            mask = th.ones(
                x.shape[0],
                device=x.device,
                dtype=th.bool
            )

            y = el_dict.pop('y')

            # we ignore all the other keys
            # remaining_keys = list(sorted(el_dict.keys()))
            # for k in remaining_keys:
                # Concatenate remaining attributes on x
            #    x = th.cat((x, el_dict[k]), dim=1)

            self.data.append(
                DenseData(
                    x,
                    adj.to(torch.float),
                    mask)
            )
            self.targets.append(y.squeeze(0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]