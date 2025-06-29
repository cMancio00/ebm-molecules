import lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10


class CIFAR10DataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = "./datasets", batch_size: int = 32, num_workers: int = 4, num_samples=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.num_classes = 10
        self.img_shape = (3, 32, 32)
        self.num_samples = num_samples

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage):
        if stage == "fit":
            mnist_full = CIFAR10(self.data_dir, train=True, download=True, transform=self.transform)
            if self.num_samples is not None:
                mnist_full.data = mnist_full.data[:self.num_samples]
                mnist_full.targets = mnist_full.targets[:self.num_samples]
            self.data_train, self.data_val = random_split(
                # TODO: Parametrize lengths from CLI
                mnist_full, [11/12, 1/12]
            )
        if stage == "test":
            self.data_test = CIFAR10(self.data_dir, train=False, download=True, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, drop_last=True, shuffle=True, pin_memory=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers)
