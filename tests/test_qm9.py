import unittest
from enum import Enum

import torch
from torch_geometric.data import Data
from assertpy import assert_that
from data_modules.QM9DataModule import densify_qm9

class OldBondType(Enum):
    SINGLE = [1, 0, 0, 0]
    DOUBLE = [0, 1, 0, 0]
    TRIPLE = [0, 0, 1, 0]
    AROMATIC = [0, 0, 0, 1]

class NewBondType(Enum):
    NONE = [1, 0, 0, 0, 0]
    SINGLE = [0, 1, 0, 0, 0]
    DOUBLE = [0, 0, 1, 0, 0]
    TRIPLE = [0 ,0, 0, 1, 0]
    AROMATIC = [0 ,0, 0, 0, 1]

class MyTestCase(unittest.TestCase):


    def setUp(self) -> None:
        # Suppose this test (invalid) molecule H--C-2-F
        # C=0,H=1,F=2
        x = torch.tensor([
        [0., 1., 0., 0., 0., 6., 0., 0., 0., 0., 1.], # C
        [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.], # H
        [0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.]]) # F
        bonds = torch.tensor([
            [0, 0, 1, 2],
            [1, 2, 0, 0]])
        bonds_type = torch.tensor([OldBondType.SINGLE.value,  # Single C-->H
                                   OldBondType.DOUBLE.value,  # Double C -->F
                                   OldBondType.SINGLE.value,  # Single H --> C
                                   OldBondType.DOUBLE.value])  # Double F --> C
        self.molecule: Data = Data(x=x, edge_index=bonds, edge_attr=bonds_type)


    def test_densify_gives_correct_bounds_encoding(self):
        dense_molecule, _ = densify_qm9(self.molecule, y=torch.tensor((1,)))
        assert_that(dense_molecule.adj[0,0].tolist()).is_equal_to(NewBondType.NONE.value)
        assert_that(dense_molecule.adj[0,1].tolist()).is_equal_to(NewBondType.SINGLE.value)
        assert_that(dense_molecule.adj[0,2].tolist()).is_equal_to(NewBondType.DOUBLE.value)
        assert_that(dense_molecule.adj[1,0].tolist()).is_equal_to(NewBondType.SINGLE.value)
        assert_that(dense_molecule.adj[1,1].tolist()).is_equal_to(NewBondType.NONE.value)
        assert_that(dense_molecule.adj[1,2].tolist()).is_equal_to(NewBondType.NONE.value)
        assert_that(dense_molecule.adj[2,0].tolist()).is_equal_to(NewBondType.DOUBLE.value)
        assert_that(dense_molecule.adj[2,1].tolist()).is_equal_to(NewBondType.NONE.value)
        assert_that(dense_molecule.adj[2,2].tolist()).is_equal_to(NewBondType.NONE.value)

if __name__ == '__main__':
    unittest.main()
