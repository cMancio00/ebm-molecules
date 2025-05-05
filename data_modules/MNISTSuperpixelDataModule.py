from typing import override, List
import lightning as pl
from torch_geometric.datasets import MNISTSuperpixels
from torch.utils.data import DataLoader
from utils.graph import dense_collate_fn, DenseGraphDataset


class MNISTSuperpixelDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = "datasets/MNISTSuperpixel",
                 batch_size: int = 32,
                 num_workers: int = 4,
                 filter_target: List[int] = range(10),
                 train_split: float = 0.9):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mnist_train = None
        self.mnist_val = None
        self.mnist_test = None
        self.filter_target = filter_target
        self.train_split = train_split

    @override
    def prepare_data(self):
        DenseGraphDataset(MNISTSuperpixels(self.data_dir, train=True))
        DenseGraphDataset(MNISTSuperpixels(self.data_dir, train=False))

    @override
    def setup(self, stage):
        if stage == "fit":
            mnist_full = DenseGraphDataset(MNISTSuperpixels(self.data_dir, train=True))
            idx = [i for i, x in enumerate(mnist_full[:][1]) if x.item() in self.filter_target]
            split = int(len(idx) * self.train_split)
            self.mnist_train = [mnist_full[i] for i in idx[:split]]
            self.mnist_val = [mnist_full[i] for i in idx[split:]]
        if stage == "test":
            # Filter as in "fit" stage
            self.mnist_test = DenseGraphDataset(MNISTSuperpixels(self.data_dir, train=False))

    @override
    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, drop_last=True, shuffle=True, pin_memory=True, collate_fn=dense_collate_fn, num_workers=self.num_workers)

    @override
    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers,
                          collate_fn=dense_collate_fn)

    @override
    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=dense_collate_fn)