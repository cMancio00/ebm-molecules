from dataclasses import dataclass
from typing import Tuple, List
import torch as th
from torch_geometric.data import Dataset as pygDataset, Batch
from torch.utils.data import Dataset
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch.nn.functional import pad

# TODO: consider store dense repr ons disk.
# TODO: it works only with a single feature channel (both on nodes and edges) -> should be true on most of the dataset
# TODO: find a way to access pygDataset attribute without breaking multiprocessing dataloader (__get_attribute__ fails)
#  Probably the best practice is to write them explicitly

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

    def __add__(self, other):
        if not isinstance(other, DenseData):
            raise ValueError("Both objects need to be of type DenseData")
        #TODO:
        #Check for concatenation conditions
        x_concat = th.cat((self.x, other.x), dim=0)
        adj_concat = th.cat((self.adj, other.adj), dim=0)
        mask_concat = th.cat((self.mask, other.mask), dim=0)
        return DenseData(x_concat, adj_concat, mask_concat)

    def __len__(self):
        return self.x.shape[0]

@dataclass
class DenseElement:
    data: DenseData
    y: th.Tensor

    def __repr__(self):
        return (
            f"DenseElement("
            f"{self.data.__repr__()}, "
            f"y={tuple(self.y.shape)})"
        )

    def __len__(self):
        return self.data.__len__()

def densify(data: Batch) -> DenseElement:
    return DenseElement(
        densify_data(data),
        data.y
    )

def densify_data(data: Batch) -> DenseData:
    x, mask = to_dense_batch(
        th.cat((data.x, data.pos), dim=1),
        data.batch
    )
    adj = to_dense_adj(data.edge_index, data.batch)
    return DenseData(x, adj, mask)

def dense_collate_fn(batch: List[Tuple[DenseData, th.Tensor]]) -> DenseElement:
    max_num_nodes = max([el[0].x.shape[0] for el in batch])
    x_list = []
    adj_list = []
    mask_list = []
    y_list = []

    for data, y in batch:
        n_nodes = data.x.shape[0]
        x = pad(data.x, (data.x.ndim - 1) * (0, 0) + (0, max_num_nodes - n_nodes)).unsqueeze(0)
        adj = pad(data.adj, (data.adj.ndim - 2) * (0, 0) + 2 * (0, max_num_nodes - n_nodes)).unsqueeze(0)
        mask = pad(data.mask, (0, max_num_nodes - n_nodes)).unsqueeze(0)

        x_list.append(x)
        adj_list.append(adj)
        mask_list.append(mask)
        y_list.append(y)

    x_stacked = th.cat(x_list, dim=0)
    adj_stacked = th.cat(adj_list, dim=0)
    mask_stacked = th.cat(mask_list, dim=0)
    y_stacked = th.cat(y_list, dim=0)

    return DenseElement(
            DenseData(x_stacked, adj_stacked, mask_stacked),
            y_stacked
    )


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
                edge_attr=el_dict.pop('edge_attr', None)
            ).squeeze(0)
            mask = th.ones(
                x.shape[0],
                device=x.device,
                dtype=th.bool
            )

            y = el_dict.pop('y')

            remaining_keys = list(sorted(el_dict.keys()))
            for k in remaining_keys:
                # Concatenate remaining attributes on x
                x = th.cat((x, el_dict[k]), dim=1)

            self.data.append(
                DenseData(
                    x,
                    adj,
                    mask)
            )
            self.targets.append(y)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]