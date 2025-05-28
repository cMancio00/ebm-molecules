from enum import Enum
from typing import List

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

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


def discretize_labels(batch: Batch) -> List[torch.Tensor]:
    """
    Given a batch of QM9 Data containing y field, returns a list of discretized
    properties using one-hot encoding with 3 classes (low, medium, high).
    The division is made by quantiles.
    """
    mol_props = []
    for prop in MoleculeProperty:
        values = batch.y[:, prop.value]
        low = torch.quantile(values, 0.25)
        high = torch.quantile(values, 0.75)
        discrete_prop: torch.Tensor = torch.ones_like(values, dtype=torch.long)
        low_mask = [values < low]
        high_mask = [values > high]

        discrete_prop[low_mask] = 0
        discrete_prop[high_mask] = 2
        discrete_prop = F.one_hot(discrete_prop, num_classes=3)
        mol_props.append(discrete_prop)
    return mol_props