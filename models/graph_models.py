from torch import nn, norm
from torch_geometric.nn import max_pool, global_mean_pool
from torch_geometric.nn.conv import GMMConv
from torch_geometric.nn.pool import graclus
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.utils import normalized_cut
import lightning as pl
import torch.optim as optim


def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))

class MoNet(pl.LightningModule):
    def __init__(self, kernel_size=3):
        super(MoNet, self).__init__()
        self.conv1 = GMMConv(1, 32, dim=2, kernel_size=kernel_size)
        self.conv2 = GMMConv(32, 64, dim=2, kernel_size=kernel_size)
        self.conv3 = GMMConv(64, 64, dim=2, kernel_size=kernel_size)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)

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
        loss = F.nll_loss(self(batch), batch.y)
        return loss