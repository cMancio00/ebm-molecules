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

class MoNet(pl.LightningModule):
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

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.97)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # loss = CrossEntropyLoss()(self(batch), batch.y)
        loss = F.nll_loss(self(batch), batch.y)
        self.log('CrossEntropy loss', loss)
        return loss

class GCN_Dense(pl.LightningModule):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.conv1 = DenseGCNConv(in_channels, hidden_channels)
        self.conv2 = DenseGCNConv(hidden_channels, hidden_channels)
        self.conv3 = DenseGCNConv(hidden_channels, hidden_channels)
        self.pool1 = DMoNPooling([hidden_channels, hidden_channels], ceil(0.5 * 75))
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x, adj, mask):
        x = self.conv1(x, adj, mask).relu()
        x = self.conv2(x, adj, mask).relu()
        x = self.conv3(x, adj, mask).relu()
        _, x, adj, _, _, _ = self.pool1(x, adj, mask)
        x = x.mean(dim=1)
        x = self.lin1(x)
        return self.lin2(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.97)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        out = self(batch.data.x, batch.data.adj, batch.data.mask)
        loss = F.cross_entropy(out, batch.y)
        self.log('CrossEntropy', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        total_correct = 0
        out = self(batch.data.x, batch.data.adj, batch.data.mask)
        pred = out.argmax(dim=-1)
        total_correct += int((pred == batch.y).sum())
        loss = total_correct / len(batch)
        self.log('Accuracy', loss)
        return loss