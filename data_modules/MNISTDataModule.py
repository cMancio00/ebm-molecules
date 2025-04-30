from typing import override
import lightning as pl
import torch
from tensorboard.data.proto.data_provider_pb2 import TensorData
from torch.utils.data import random_split, DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = "./datasets", batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.mnist_train = None
        self.mnist_val = None
        self.mnist_test = None
        self.num_classes = 10
        self.img_shape = (1, 28, 28)

    @override
    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    @override
    def setup(self, stage):
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, download=True, transform=self.transform)
            #to_take = mnist_full.targets < 2
            #mnist_full.data = mnist_full.data[to_take]
            #mnist_full.targets = mnist_full.targets[to_take]
            self.mnist_train, self.mnist_val = random_split(
                # TODO: Parametrize lengths from CLI
                mnist_full, [11/12, 1/12]
            )
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, download=True, transform=self.transform)

    @override
    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, drop_last=True, shuffle=True, pin_memory=True,
                          num_workers=self.num_workers)

    @override
    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    @override
    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)
