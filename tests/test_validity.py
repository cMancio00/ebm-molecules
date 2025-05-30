import torch
from rdkit import Chem
from tqdm import tqdm

from data_modules import QM9DataModule
from utils.mol import make_valid, _correct_mol


def test_correct_mol():
    mol = Chem.RWMol()
    mol.AddAtom(Chem.Atom(6))
    mol.AddAtom(Chem.Atom(6))
    mol.AddAtom(Chem.Atom(6))
    mol.AddAtom(Chem.Atom(7))
    mol.AddBond(0, 1, Chem.rdchem.BondType.DOUBLE)
    mol.AddBond(1, 2, Chem.rdchem.BondType.TRIPLE)
    mol.AddBond(0, 3, Chem.rdchem.BondType.TRIPLE)
    print(Chem.MolToSmiles(mol))  # C#C=C#N
    mol = _correct_mol(mol)
    print(Chem.MolToSmiles(mol))  # C=C=C=N


def test_validity():
    ROOT = "./datasets/QM9"
    dm = QM9DataModule(data_dir=ROOT, batch_size=200)
    dm.prepare_data()
    dm.setup("fit")
    t = dm.train_dataloader()
    for b in tqdm(t):
        data, y = b
        x = data.x
        #print(x[0][0])
        x[0][0] = torch.tensor([0., 0., 0., 1.])
        x[0][1] = torch.tensor([0., 0., 0., 1.])
        x[0][2] = torch.tensor([0., 0., 0., 1.])
        data.x = x

        make_valid(data)