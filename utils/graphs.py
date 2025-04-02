import itertools
import torch
from torch_geometric.data import Data, Batch
import numpy as np
import cv2
from torch_geometric.utils import dense_to_sparse
from torch import device

def concat_batches(batches: list[Batch]) -> Batch:
    return Batch.from_data_list(
        list(itertools.chain(*[batch.to_data_list() for batch in batches]))
    )

def generate_random_graph(n_nodes: int = 75, n_edge_feature: int = 2, device: device = device('cpu')) -> Data:
    x: torch.Tensor = torch.rand((n_nodes, 1), device=device, requires_grad=True)
    edge_index: torch.Tensor = dense_to_sparse(
        torch.rand((n_nodes, n_nodes), dtype=torch.float32, device=device, requires_grad=True))[0]
    pos: torch.Tensor = torch.rand((n_nodes, 2), device=device) * 28
    edge_attr: torch.Tensor = torch.rand((edge_index.shape[1], n_edge_feature), device=device)
    return Data(x=x, edge_index=edge_index, pos=pos, edge_attr=edge_attr)

def superpixels_to_2d_image(rec: Data, scale: int = 30, edge_width: int = 1) -> np.ndarray:
    pos = (rec.pos.clone() * scale).int()

    image = np.zeros((scale * 26, scale * 26, 1), dtype=np.uint8)
    for (color, (x, y)) in zip(rec.x, pos):
        x0, y0 = int(x), int(y)
        x1, y1 = x0 - scale, y0 - scale

        color = int(float(color + 0.15) * 255)
        color = min(color, 255)

        cv2.rectangle(image, (x0, y0), (x1, y1), color, -1)

    for node_ix_0, node_ix_1 in rec.edge_index.T:
        x0, y0 = list(map(int, pos[node_ix_0]))
        x1, y1 = list(map(int, pos[node_ix_1]))

        x0 -= scale // 2
        y0 -= scale // 2
        x1 -= scale // 2
        y1 -= scale // 2

        cv2.line(image, (x0, y0), (x1, y1), 125, edge_width)
    return image