from typing import Tuple, List, Sequence
import torch as th
from torch_geometric.data import Dataset as pygDataset
from torch.utils.data import DataLoader, Dataset
from torch_geometric.utils import to_dense_adj
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
from collections import namedtuple

# TODO: consider store dense repr ons disk.
# TODO: it works only with a single feature channel (both on nodes and edges) -> should be true on most of the dataset


def dense_collate_fn(batch: List[Sequence[th.Tensor]]):
    num_attr = len(batch[0])
    max_num_nodes = max([el[0].shape[0] for el in batch])
    zipped_padded_batch = [[] for _ in range(num_attr)]
    for el in batch:
        n_nodes = el[0].shape[0]
        for i, attr in enumerate(el):
            n_dim = attr.ndim
            if i == 0:
                # this is the node feature x, padding on first dimension
                padded_attr = pad(attr, (n_dim-1)*(0, 0) + (0, max_num_nodes - n_nodes)).unsqueeze(0)
            elif i == 1:
                # this is the dense adj, padding on the first two dimension
                padded_attr = pad(attr, (n_dim-2)*(0, 0) + 2*(0, max_num_nodes - n_nodes)).unsqueeze(0)
            elif i == 2:
                # this is the node_mask, padding on the first (and only) dimension
                padded_attr = pad(attr, (0, max_num_nodes - n_nodes)).unsqueeze(0)
            else:
                # these are other attributes: we always assume are nodes features or scalars
                if attr.ndim == 0 or attr.shape[0] == 1:
                    # this is a scalar, no need of padding
                    padded_attr = attr
                elif attr.ndim < 3:
                    # node attributes
                    padded_attr = pad(attr, (n_dim-1)*(0, 0) + (0, max_num_nodes - n_nodes)).unsqueeze(0)
                else:
                    raise ValueError(f'Unsupported attribute with shape {attr.shape}')
            zipped_padded_batch[i].append(padded_attr)

    return (type(batch[0]))(*[th.cat(zipped_padded_batch[i], dim=0) for i in range(len(batch[0]))])


class DenseGraphDataset(Dataset):

    """
    This class is wrapper of a pygDataset.
    """

    @staticmethod
    def __build_dense_element_type(attr_names):
        class DenseElement(namedtuple('BaseDenseElement', attr_names)):
            def __str__(self):
                s = f'{self.__class__.__name__}('
                for attr_name in attr_names:
                    s += f'{attr_name}={getattr(self, attr_name).shape}, '
                s = s[:-2] + ')'
                return s

            def __repr__(self):
                return self.__str__()

        return DenseElement

    def __init__(self, pyg_dataset: pygDataset):
        self._pyg_dataset = pyg_dataset
        self._ordered_attr_names = ['x', 'adj', 'node_mask']
        for k in self._pyg_dataset[0].to_dict():
            if k not in ['edge_index', 'edge_attr', 'x']:
                self._ordered_attr_names.append(k)

        self._element_type = self.__build_dense_element_type(self._ordered_attr_names)
        self._data_list = []
        for el in self._pyg_dataset:
            adj = to_dense_adj(el.edge_index, edge_attr=el.edge_attr).squeeze(0)

            el_tuple = [el.x, adj, th.ones((el.x.shape[0]), device=el.x.device, dtype=th.bool)]
            for k in self._ordered_attr_names[3:]:
                el_tuple.append(getattr(el, k))
            self._data_list.append(self._element_type(*el_tuple))

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, idx):
        return self._data_list[idx]

    def __getattr__(self, item):
        return getattr(self._pyg_dataset, item)


