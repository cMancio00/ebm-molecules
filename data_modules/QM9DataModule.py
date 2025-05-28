from typing import Tuple
import lightning as pl
from torch_geometric.datasets import QM9
import torch
from torch.utils.data import DataLoader, random_split
from utils.graph import dense_collate_fn, DenseData


class QM9DataModule(pl.LightningDataModule):

    MAX_SAMPLES = 130831

    def __init__(self, data_dir: str =  "./datasets/QM9/",
                 batch_size: int = 32,
                 num_workers: int = 4,
                 num_samples: int = MAX_SAMPLES):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        if num_samples >= self.MAX_SAMPLES:
            self.num_samples = self.MAX_SAMPLES
        else:
            self.num_samples = num_samples

        self.data_train = None
        self.data_val = None
        self.data_test = None


    def prepare_data(self):
        QM9(root=self.data_dir)

    def setup(self, stage):
        #TODO: See if we can filter and transform before
        dataset_full = QM9(root=self.data_dir)
        idx = torch.randint(0, self.MAX_SAMPLES, (self.num_samples,))
        dataset = []
        for data in dataset_full[idx]:
            dataset.append(densify_qm9(data))
        self.data_train, self.data_val, self.data_test = random_split(dataset, [0.7, 0.2, 0.1])

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, drop_last=True, shuffle=True, pin_memory=True,
                          collate_fn=dense_collate_fn, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers,
                          collate_fn=dense_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers,
                          collate_fn=dense_collate_fn)

def densify_qm9(data) -> Tuple[DenseData, torch.Tensor]:
    x = data.x[:,:-6]
    x_dim = x.size(0)
    adj_3d = torch.zeros((x_dim, x_dim, 5))

    # NO BOUNDS = [1.,0.,0.,0.,0.]
    adj_3d[:,:,0] = 1

    bonds = data.edge_index
    bonds_type = data.edge_attr
    src = bonds[0]
    des = bonds[1]
    type_ = torch.argmax(bonds_type, dim=1)
    adj_3d[src, des, type_ + 1] = 1
    adj_3d[src, des, 0] = 0

    mask = torch.ones(
        x.shape[0],
        device=x.device,
        dtype=torch.bool
    )
    return DenseData(x=x, adj=adj_3d, mask=mask), torch.ones((1,), device=x.device)

