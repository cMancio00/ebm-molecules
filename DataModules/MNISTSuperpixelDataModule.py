from pathlib import Path
from typing import override
import lightning as pl
from torch.utils.data import random_split
from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T
from utils.data import DenseGraphDataset, dense_collate_fn
from torch.utils.data import DataLoader


class MNISTSuperpixelDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = "datasets/MNISTSuperpixel", batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mnist_train = None
        self.mnist_val = None
        self.mnist_test = None

    @override
    def prepare_data(self):
        DenseGraphDataset(MNISTSuperpixels(self.data_dir, train=True))
        DenseGraphDataset(MNISTSuperpixels(self.data_dir, train=False))

    @override
    def setup(self, stage):
        if stage == "fit":
            mnist_full = DenseGraphDataset(MNISTSuperpixels(self.data_dir, train=True))
            to_take = (mnist_full.y == 0) | (mnist_full.y == 1)
            mnist_full = mnist_full[to_take]

            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [11/12, 1/12]
            )
        if stage == "test":
            self.mnist_test = DenseGraphDataset(MNISTSuperpixels(self.data_dir, train=False))

    @override
    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, drop_last=True, shuffle=True, pin_memory=True,
                          num_workers=self.num_workers, collate_fn=dense_collate_fn)

    @override
    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers,
                          collate_fn=dense_collate_fn)

    @override
    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers,
                          collate_fn=dense_collate_fn)
