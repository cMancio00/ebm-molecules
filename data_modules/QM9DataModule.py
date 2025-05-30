import sys
from enum import Enum
from typing import Tuple, List
import lightning as pl
from rdkit import Chem
from torch_geometric.datasets import QM9
from torch_geometric.data import Data
import torch
from torch.utils.data import DataLoader, random_split
from utils.graph import dense_collate_fn, DenseData, DenseGraphDataset
import torch.nn.functional as F
from utils.mol import from_rdkit_mol, densify_mol
from tqdm import tqdm
from rdkit import RDLogger


class MoleculeProperty(Enum):
    DIPOLE_MOMENT = 0
    ISOTROPIC_POLARIZABILITY = 1
    HOMO_ENERGY = 2
    LUMO_ENERGY = 3
    ENERGY_GAP = 4
    ELECTRONIC_SPATIAL_EXTENT = 5
    ZERO_POINT_VIBRATIONAL_ENERGY = 6
    INTERNAL_ENERGY_AT_0K = 7
    INTERNAL_ENERGY_AT_298K = 8
    ENTHALPY_AT_298K = 9
    FREE_ENERGY_AT_298K = 10
    HEAT_CAPACITY_AT_298K = 11
    ATOMIZATION_ENERGY_AT_0K = 12
    ATOMIZATION_ENERGY_AT_298K = 13
    ATOMIZATION_ENTHALPY_AT_298K = 14
    ATOMIZATION_FREE_ENERGY_AT_298K = 15
    ROTATIONAL_CONSTANT_A = 16
    ROTATIONAL_CONSTANT_B = 17
    ROTATIONAL_CONSTANT_C = 18


class MyQM9Datasets(QM9):

    HAR2EV = 27.211386246
    KCALMOL2EV = 0.04336414

    conversion = torch.tensor([
        1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
        1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
    ])

    def process(self) -> None:
        RDLogger.DisableLog('rdApp.*')
        with open(self.raw_paths[1], 'r') as f:
            target = [[float(x) for x in line.split(',')[1:20]]
                      for line in f.read().split('\n')[1:-1]]
            y = torch.tensor(target, dtype=torch.float)
            y = torch.cat([y[:, 3:], y[:, :3]], dim=-1)
            y = y * self.conversion.view(1, -1)

        with open(self.raw_paths[2], 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=False)

        data_list = []
        tot = 0
        for i, mol in enumerate(tqdm(suppl)):
            tot += 1
            if i in skip:
                continue

            data = from_rdkit_mol(mol)
            if data is None:
                continue

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data.update({'y': y[i].unsqueeze(0), 'idx': i})

            data_list.append(data)
        print(f'Process {len(data_list)} molecules over {tot}!', file=sys.stderr)

        self.save(data_list, self.processed_paths[0])


class QM9DataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = "./datasets/QM9/",
                 batch_size: int = 32,
                 num_workers: int = 4,
                 num_samples: int = None,
                 properties: List[int] = None):

        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.num_samples = num_samples

        if properties is None:
            self.properties = [
                     MoleculeProperty.DIPOLE_MOMENT.value,
                     MoleculeProperty.HOMO_ENERGY.value
                 ]
        else:
            self.properties = properties

        self.data_train = None
        self.data_val = None
        self.data_test = None

        self.quantile = None

    def prepare_data(self):
        MyQM9Datasets(root=self.data_dir)

    def __get_discrete_label(self, data) -> torch.Tensor:
        """
        Given a batch of QM9 Data containing y field, returns a list of discretized
        properties using one-hot encoding with 3 classes (low, medium, high).
        The division is made by quantiles.
        """
        mol_props = []
        value = data.y[:, self.properties]
        low = self.quantile[0][self.properties]
        high = self.quantile[1][self.properties]
        discrete_prop: torch.Tensor = torch.ones_like(value, dtype=torch.long)
        low_mask = [value < low]
        high_mask = [value > high]

        discrete_prop[low_mask] = 0
        discrete_prop[high_mask] = 2
        discrete_prop = F.one_hot(discrete_prop, num_classes=3).transpose(-1, -2)
        mol_props.append(discrete_prop)
        return torch.cat(mol_props, dim=0)

    def setup(self, stage):
        dataset_full = MyQM9Datasets(root=self.data_dir)
        # compute quantile
        all_y = dataset_full.y
        self.quantile = [torch.quantile(all_y, q=0.25, dim=0), torch.quantile(all_y, q=0.75, dim=0)]

        if self.num_samples is not None and self.num_samples < len(dataset_full):
            idx = torch.randperm(len(dataset_full))[:self.num_samples]
            dataset_full = dataset_full[idx]

        dataset_full = DenseGraphDataset(dataset_full,
                                         get_dense_data_fun=densify_mol,
                                         get_y_fun=self.__get_discrete_label)

        self.data_train, self.data_val, self.data_test = random_split(dataset_full, [0.7, 0.2, 0.1])

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, drop_last=True, shuffle=True, pin_memory=True,
                          collate_fn=dense_collate_fn, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers,
                          collate_fn=dense_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=self.num_workers,
                          collate_fn=dense_collate_fn)


if __name__ == '__main__':
    dm = QM9DataModule(data_dir='../datasets', num_workers=0)
    dm.prepare_data()
    dm.setup('fit')
    tr_loader = dm.train_dataloader()
    for el in tr_loader:
        pass

