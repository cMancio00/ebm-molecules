import warnings
from typing import Tuple

import rdkit
import torch
from rdkit import Chem
from .graph import DenseData
import rdkit
from rdkit.Chem.rdchem import BondType as BT
from rdkit import Chem
from torch_geometric.utils import one_hot, scatter
from torch_geometric.data import Data


ATOM_TYPE = {n: i for i, n in enumerate(['C', 'N', 'O', 'F'])}
BOND_TYPE = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}


def from_rdkit_mol(mol: rdkit.Chem.Mol) -> Data:

    mol = Chem.RemoveAllHs(mol, False)
    ris = Chem.SanitizeMol(mol, Chem.SANITIZE_ALL, True)
    if ris > 0:
        # an error occured (probably valence error)
        return None

    Chem.Kekulize(mol)

    N = mol.GetNumAtoms()

    type_idx = []

    for atom in mol.GetAtoms():
        type_idx.append(ATOM_TYPE[atom.GetSymbol()])

    rows, cols, edge_types = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rows += [start, end]
        cols += [end, start]
        edge_types += 2 * [BOND_TYPE[bond.GetBondType()]]

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_attr = torch.tensor(edge_types, dtype=torch.long)
    edge_attr = one_hot(edge_attr, num_classes=len(BOND_TYPE))

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]

    x = one_hot(torch.tensor(type_idx), num_classes=len(ATOM_TYPE))

    name = mol.GetProp('_Name')
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

    data = Data(
        x=x,
        edge_index=edge_index,
        smiles=smiles,
        edge_attr=edge_attr,
        name=name,
    )
    return data


def densify_molecule(data: Data) -> DenseData:
    x = data.x
    x_dim = x.size(0)
    adj_3d = torch.zeros((x_dim, x_dim, len(BOND_TYPE)))

    # NO BOUNDS = [1.,0.,0.,0.,0.]
    adj_3d[:, :, 0] = 1

    bonds = data.edge_index
    bonds_type = data.edge_attr
    src = bonds[0]
    des = bonds[1]
    type_ = torch.argmax(bonds_type, dim=1)
    adj_3d[src, des, type_ + 1] = 1
    adj_3d[src, des, 0] = 0

    mask = torch.ones(
        x.shape[0],
        device=x.device,
        dtype=torch.bool
    )
    return DenseData(x=x, adj=adj_3d, mask=mask)

