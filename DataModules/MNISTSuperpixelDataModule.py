from dataclasses import dataclass
from typing import override, Iterator, Any
import lightning as pl
import torch
from torch.utils.data import random_split
from torch_geometric.data import Dataset, Batch
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.loader import DataLoader

@dataclass
class DenseData:
    x: torch.Tensor
    adj: torch.Tensor
    mask: torch.Tensor

    def __repr__(self):
        return (
            f"DenseData("
            f"x={tuple(self.x.shape)}, "
            f"adj={tuple(self.adj.shape)}, "
            f"mask={tuple(self.mask.shape)})"
        )

    def __add__(self, other):
        if not isinstance(other, DenseData):
            raise ValueError("Both objects need to be of type DenseData")
        #TODO:
        #Check for concatenation conditions
        x_concat = torch.cat((self.x, other.x), dim=0)
        adj_concat = torch.cat((self.adj, other.adj), dim=0)
        mask_concat = torch.cat((self.mask, other.mask), dim=0)
        return DenseData(x_concat, adj_concat, mask_concat)

    def __getitem__(self, index):
        return DenseData(
            self.x[index],
            self.adj[index],
            self.mask[index]
        )

    def __len__(self):
        return self.x.shape[0]


@dataclass
class DenseElement:
    data: DenseData
    y: torch.Tensor

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
        torch.cat((data.x, data.pos), dim=1),
        data.batch
    )
    adj = to_dense_adj(data.edge_index, data.batch)
    return DenseData(x, adj, mask)


class DenseMNISTDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        **kwargs,
    ):
        kwargs.pop('collate_fn', None)

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs,
        )

    def __iter__(self) -> Iterator[DenseElement]:
        for batch in super().__iter__():
            yield densify(batch)


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
        MNISTSuperpixels(self.data_dir, train=True)
        MNISTSuperpixels(self.data_dir, train=False)

    @override
    def setup(self, stage):
        if stage == "fit":
            mnist_full = MNISTSuperpixels(self.data_dir, train=True)
            to_take = (mnist_full.y == 0) | (mnist_full.y == 1)
            mnist_full = mnist_full[to_take]

            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [11/12, 1/12]
            )
        if stage == "test":
            self.mnist_test = MNISTSuperpixels(self.data_dir, train=False)

    @override
    def train_dataloader(self):
        return DenseMNISTDataLoader(self.mnist_train, batch_size=self.batch_size, drop_last=True, shuffle=True, pin_memory=True,
                          num_workers=self.num_workers)

    @override
    def val_dataloader(self):
        return DenseMNISTDataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    @override
    def test_dataloader(self):
        return DenseMNISTDataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)