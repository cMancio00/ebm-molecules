from typing import List, Union, Tuple
import lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from utils.graph import dense_collate_fn, DenseGraphDataset
import torch_geometric.utils as geom_utils
from torch_geometric.data import Data, InMemoryDataset
import os.path as osp


class SBMDataset(InMemoryDataset):
    NUM_GRAPHS = None
    AVG_NUM_NODES = None
    NUM_CLASSES = None
    P_INTRA_CLASS = None
    P_EXTRA_CLASS = None
    FOLDER_NAME = None
    NUM_NODE_FEATURES = 1
    NUM_EDGE_FEATURES = 1

    def __init__(self, root, transform=None, pre_transform=None):
        root = osp.join(root, f'{self.FOLDER_NAME}')
        super(SBMDataset, self).__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        return ['data.pt']

    def process(self) -> None:
        data_list = []
        n_graphs_for_class = self.NUM_GRAPHS // self.NUM_CLASSES
        for i in range(self.NUM_CLASSES):
            num_blocks = i + 1
            edge_probs = self.P_INTRA_CLASS * torch.eye(num_blocks) + self.P_EXTRA_CLASS * (1 - torch.eye(num_blocks))
            for _ in range(n_graphs_for_class):
                n_nodes = int(self.AVG_NUM_NODES + 2 * torch.randn(1).item())
                n_nodes = max(n_nodes, 5)
                block_sizes = (n_nodes // num_blocks * torch.ones(num_blocks)).to(torch.long)
                n_nodes = torch.sum(block_sizes).item()
                edge_index = geom_utils.stochastic_blockmodel_graph(block_sizes, edge_probs)
                edge_index, _ = geom_utils.remove_self_loops(edge_index)
                node_community = torch.cat([block_sizes.new_full((b,), i) for i, b in enumerate(block_sizes)])
                x = torch.randn((n_nodes, self.NUM_NODE_FEATURES)) + 5 * node_community.view(-1, 1)

                # remove isolated nodes
                edge_index, _, mask = geom_utils.remove_isolated_nodes(edge_index, num_nodes=n_nodes)
                x = x[mask]

                data_list.append(Data(x=x, edge_index=edge_index, y=i))

        self.save(data_list, self.processed_paths[0])


class SBMDatasetEasy(SBMDataset):

    NUM_GRAPHS = 20000
    AVG_NUM_NODES = 20
    NUM_CLASSES = 2
    P_INTRA_CLASS = 0.9
    P_EXTRA_CLASS = 0.1
    FOLDER_NAME = 'easy'


class SBMDataModule(pl.LightningDataModule):

    CLASS_NAME_DICT = {'easy': SBMDatasetEasy}

    def __init__(self, data_dir: str = "datasets",
                 name='SBM/easy',
                 batch_size: int = 32,
                 num_workers: int = 4):

        super().__init__()
        self.data_dir = data_dir
        self.class_name = self.CLASS_NAME_DICT[name]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_train = None
        self.data_val = None
        self.data_test = None

    @property
    def num_classes(self) -> int:
        return self.class_name.NUM_CLASSES

    @property
    def num_edge_features(self) -> int:
        return self.class_name.NUM_EDGE_FEATURES

    @property
    def num_node_features(self) -> int:
        return self.class_name.NUM_NODE_FEATURES

    def _get_full_dataset(self):
        return DenseGraphDataset(self.class_name(self.data_dir))

    def prepare_data(self):
        self._get_full_dataset()

    def setup(self, stage):
        dataset_full = self._get_full_dataset()
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
    dm = SBMDataModule(data_dir='../datasets', num_workers=0)
    dm.prepare_data()
    dm.setup('fit')
    tr_loader = dm.train_dataloader()
    for el in tr_loader:
        pass

