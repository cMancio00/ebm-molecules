import itertools
from typing import Tuple, List
import torch
from torch_geometric.data import Data, Batch
import numpy as np
import cv2
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import to_dense_batch, to_dense_adj

from utils.data import DenseData


def concat_batches(batches: list[Batch]) -> Batch:
    return Batch.from_data_list(
        list(itertools.chain(*[batch.to_data_list() for batch in batches]))
    )

def generate_random_graph(num_nodes: int = 75, num_edges: int = 1500, device: torch.device = torch.device('cpu')) -> Data:
    edges: torch.Tensor = torch.randint(0, num_nodes, (num_edges, 2), dtype=torch.long)
    x: torch.Tensor = torch.rand((num_nodes, 1))
    pos: torch.Tensor = torch.rand((num_nodes, 2)) * 28
    return Data(x=x, edge_index=edges.t().contiguous(), pos=pos).coalesce().to(device)

def densify(data: Batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x, mask = to_dense_batch(data.x, data.batch)
    adj = to_dense_adj(data.edge_index, data.batch)
    return x, adj, mask

def to_sparse_list(x, adj, mask, ptr) -> List[Data]:
    # Maybe it is possible to not use ptr
    data = []
    for i in range(1, len(ptr)):
        sparse_x = x[mask][ptr[i-1]:ptr[i]]
        edge_index = dense_to_sparse(adj[i-1], mask)[0]
        data.append(Data(x=sparse_x, edge_index=edge_index))
    return data


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