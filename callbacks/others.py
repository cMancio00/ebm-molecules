from typing import Set

import lightning as pl
from lightning import Trainer, LightningModule
from rdkit import Chem
from torch.nn.utils.parametrizations import spectral_norm
from torch.nn.utils.parametrize import is_parametrized

from utils.mol import to_rdkit_mol
from samplers import MolSampler


class SpectralNormalizationCallback(pl.Callback):

    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        for module in pl_module.nn_model.modules():
            if hasattr(module, "weight") and ("weight" in dict(module.named_parameters())):
                if not is_parametrized(module, "weight"):
                    spectral_norm(module, name="weight", n_power_iterations=1)



class ComputeSmilesOnTrainingDatasetCallback(pl.Callback):
    def __init__(self):
        super().__init__()

    def _compute_smiles(self, train_dataloader) -> Set:
        smiles = set()
        for data, _ in iter(train_dataloader):
            batch_size = data.x.shape[0]
            for i in range(batch_size):
                x = data[i].x
                adj = data[i].adj
                mask = data[i].mask

                mol = to_rdkit_mol(x, adj, mask)
                smiles.add(Chem.MolToSmiles(mol, isomericSmiles=True))
        return smiles

    def on_train_start(self, trainer: Trainer, model: LightningModule):
        if not isinstance(model.sampler, MolSampler):
            raise ValueError("Only MolSampler can have SMILES set")

        model.sampler.smile_set = self._compute_smiles(trainer.train_dataloader)
