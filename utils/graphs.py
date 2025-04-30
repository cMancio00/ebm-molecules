import itertools
from dataclasses import dataclass
from typing import Tuple, List
import torch
import torch as th
from torch._C._nn import pad
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch, Dataset as pygDataset
import numpy as np
import cv2
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch_geometric.utils import to_dense_batch, to_dense_adj

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

    def __add__(self, other):
        if not isinstance(other, DenseData):
            raise ValueError("Both objects need to be of type DenseData")
        #TODO: check for concatenation conditions
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

            # we ignore all the other keys
            # remaining_keys = list(sorted(el_dict.keys()))
            # for k in remaining_keys:
                # Concatenate remaining attributes on x
            #    x = th.cat((x, el_dict[k]), dim=1)

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

def superpixels_to_image(rec: DenseData, scale: int = 30, edge_width: int = 1) -> np.ndarray:
    pos = (rec.x[:,1:].clone() * scale).int()

    image = np.zeros((scale * 26, scale * 26, 1), dtype=np.uint8)
    for (color, (x, y)) in zip(rec.x[:,0], pos):
        x0, y0 = int(x), int(y)
        x1, y1 = x0 - scale, y0 - scale

        color = int(float(color + 0.15) * 255)
        color = min(color, 255)

        cv2.rectangle(image, (x0, y0), (x1, y1), color, -1)

    edge_index = dense_to_sparse(rec.adj)[0]
    for node_ix_0, node_ix_1 in edge_index.T:
        x0, y0 = list(map(int, pos[node_ix_0]))
        x1, y1 = list(map(int, pos[node_ix_1]))

        x0 -= scale // 2
        y0 -= scale // 2
        x1 -= scale // 2
        y1 -= scale // 2

        cv2.line(image, (x0, y0), (x1, y1), 125, edge_width)
    return image

