import re
from typing import List, Union
import torch
from rdkit import Chem, RDLogger
from .graph import DenseData, dense_collate_fn
import rdkit
from rdkit.Chem.rdchem import BondType as BT
from rdkit import Chem
from torch_geometric.utils import one_hot, scatter
from torch_geometric.data import Data


# TODO: find a better way to handle the atoms and bonds type for each dataset
ATOM_VALENCY = {6:4, 7:3, 8:2, 9:1, 15:3, 16:2, 17:1, 35:1, 53:1}
ATOM_TYPE_2_ID = {n: i for i, n in enumerate(['C', 'N', 'O', 'F'])}
ID_2_ATOM_TYPE = {i: n for i, n in enumerate(['C', 'N', 'O', 'F'])}

BOND_TYPE_2_ID = {n: i for i, n in enumerate([BT.SINGLE, BT.DOUBLE, BT.TRIPLE])}
ID_2_BOND_TYPE = {i: n for i, n in enumerate([BT.SINGLE, BT.DOUBLE, BT.TRIPLE])}


def from_rdkit_mol(mol: rdkit.Chem.Mol) -> Union[Data, None]:

    mol = Chem.RemoveAllHs(mol, False)
    ris = Chem.SanitizeMol(mol, Chem.SANITIZE_ALL, True)
    if ris > 0:
        # an error occured (probably valence error)
        return None

    Chem.Kekulize(mol)

    N = mol.GetNumAtoms()

    type_idx = []

    for atom in mol.GetAtoms():
        type_idx.append(ATOM_TYPE_2_ID[atom.GetSymbol()])

    rows, cols, edge_types = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rows += [start, end]
        cols += [end, start]
        edge_types += 2 * [BOND_TYPE_2_ID[bond.GetBondType()]]

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_attr = torch.tensor(edge_types, dtype=torch.long)
    edge_attr = one_hot(edge_attr, num_classes=len(BOND_TYPE_2_ID))

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]

    x = one_hot(torch.tensor(type_idx), num_classes=len(ATOM_TYPE_2_ID))


    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

    data = Data(
        x=x,
        edge_index=edge_index,
        smiles=smiles,
        edge_attr=edge_attr
    )

    return data


def densify_mol(data: Data) -> DenseData:
    x = data.x
    x_dim = x.size(0)
    adj_3d = torch.zeros((x_dim, x_dim, len(BOND_TYPE_2_ID)+1))

    # NO BOUNDS = [1.,0.,0.,0.]
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


def make_valid(data: DenseData) -> List[DenseData]:
    """
     Make the mol valid
    :param data: the molecule represented as graf
    :return:
    """
    mol_list = to_rdkit_mol(data)
    valid_data_list = []
    for mol in mol_list:
        # Chem.Kekulize(mol, clearAromaticFlags=True)
        cmol = _correct_mol(mol)
        vcmol = _valid_mol_can_with_seg(cmol, largest_connected_comp=True)
        dense_data = densify_mol(from_rdkit_mol(vcmol))
        valid_data_list.append(dense_data)

    return valid_data_list


def to_rdkit_mol(data: DenseData) -> List[Chem.Mol]:
    """
    :return:
    """
    x_batch, adj_batch, mask_batch = data.x, data.adj, data.mask
    # x has shape BS x N x N_ATOMS
    # adj has shape BS x N x N x N_BONDS
    # mask has shape BS x N
    x_batch = torch.argmax(x_batch, dim=-1)
    adj_batch = torch.argmax(adj_batch, dim=-1)

    BS = x_batch.shape[0]
    out_mol_list = []

    for i in range(BS):
        mol = Chem.RWMol()
        atoms = x_batch[i, mask_batch[i]]
        N = atoms.shape[0]

        for id in range(N):
            atom_id = atoms[id].item()
            mol.AddAtom(Chem.Atom(ID_2_ATOM_TYPE[atom_id]))

        # A (edge_type, num_node, num_node)
        adj = adj_batch[i, mask_batch[i], :][:, mask_batch[i]]

        for start in range(N):
            for end in range(N):
                if start > end and adj[start, end] != 0:
                    bond_id = adj[start, end].item() - 1
                    mol.AddBond(start, end, ID_2_BOND_TYPE[bond_id])
                    # add formal charge to atom: e.g. [O+], [N+] [S+]
                    # not support [O-], [N-] [S-]  [NH+] etc.
                    flag, atomid_valence = _check_valency(mol)
                    if flag:
                        continue
                    else:
                        assert len(atomid_valence) == 2
                        idx = atomid_valence[0]
                        v = atomid_valence[1]
                        an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                        if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                            mol.GetAtomWithIdx(idx).SetFormalCharge(1)
        out_mol_list.append(mol)

    return out_mol_list


def _valid_mol_can_with_seg(x, largest_connected_comp=True):
    # mol = None
    if x is None:
        return None
    sm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = Chem.MolFromSmiles(sm)
    if largest_connected_comp and '.' in sm:
        vsm = [(s, len(s)) for s in sm.split('.')]  # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    return mol


def _check_valency(mol):
    """
    Checks that no atoms in the mol have exceeded their possible
    valency
    :return: True if no valency issues, False otherwise
    """
    RDLogger.DisableLog('rdApp.*')
    try:

        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        out = (True, None)

    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        out = (False, atomid_valence)
    finally:
        RDLogger.EnableLog('rdApp.*')

    return out


def _correct_mol(mol: Chem.Mol):
    # xsm = Chem.MolToSmiles(mol, isomericSmiles=True)
    while True:
        flag, atomid_valence = _check_valency(mol)
        if flag:
            break
        else:
            assert len (atomid_valence) == 2
            idx = atomid_valence[0]
            v = atomid_valence[1]
            queue = []
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                queue.append(
                    (b.GetIdx(), b.GetBondType(), b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                )
            queue.sort(key=lambda tup: int(tup[1]), reverse=True)
            if len(queue) > 0:
                start = queue[0][2]
                end = queue[0][3]
                bt = queue[0][1]
                mol.RemoveBond(start, end)
                if bt == BT.TRIPLE:
                    mol.AddBond(start, end, BT.DOUBLE)
                elif bt == BT.DOUBLE:
                    mol.AddBond(start, end, BT.SINGLE)
                elif bt == BT.SINGLE:
                    pass
                else:
                    raise ValueError('Unknown bond type')

    return mol
