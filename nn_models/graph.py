from torch import nn, norm
from torch_geometric.nn import Linear, DMoNPooling
from torch_geometric.nn.dense import DenseGCNConv
from torch_geometric.nn import max_pool, global_mean_pool
from torch_geometric.nn.conv import GMMConv
from torch_geometric.nn.pool import graclus
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.utils import normalized_cut
import lightning as pl
import torch.optim as optim
from math import ceil


def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))

class MoNet(nn.Module):
    def __init__(self, kernel_size: int =3, out_dim: int = 10):
        super(MoNet, self).__init__()
        self.conv1 = GMMConv(1, 32, dim=2, kernel_size=kernel_size)
        self.conv2 = GMMConv(32, 64, dim=2, kernel_size=kernel_size)
        self.conv3 = GMMConv(64, 64, dim=2, kernel_size=kernel_size)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data.x = F.elu(self.conv3(data.x, data.edge_index, data.edge_attr))

        x = global_mean_pool(data.x, data.batch)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)

class GCN_Dense(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.conv1 = DenseGCNConv(in_channels, hidden_channels)
        self.conv2 = DenseGCNConv(hidden_channels, hidden_channels)
        self.conv3 = DenseGCNConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, in_data):
        x, adj, mask = in_data.x, in_data.adj, in_data.mask
        x = self.conv1(x, adj, mask).relu()
        x = self.conv2(x, adj, mask).relu()
        x = self.conv3(x, adj, mask).relu()

        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        return self.lin2(x)