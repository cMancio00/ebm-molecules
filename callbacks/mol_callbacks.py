from lightning import Callback
import lightning as pl
from metrics.mol_metrics import Uniqueness, Novelty
from samplers import MolSampler


class ComputeMolMetricsCallback(Callback):
    def __init__(self, mol_to_generate: int = 5000):
        super().__init__()
        self.mol_to_generate = mol_to_generate

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        mols = pl_module.generate_samples(self.mol_to_generate)

        if not isinstance(pl_module.sampler, MolSampler):
            raise ValueError(f"{pl_module.sampler} is not a MolSampler")
        uniqueness = Uniqueness()
        uniqueness.update(generated_batch=mols, batch_size=self.mol_to_generate)
        # trainer.logger.experiment.add_scalar("uniqueness/test", uniqueness.compute())
        pl_module.log("uniqueness/test", uniqueness.compute())

        novelty = Novelty()
        novelty.update(generated_batch=mols, training_smiles=trainer.datamodule.training_smiles, batch_size=self.mol_to_generate)
        pl_module.log("novelty/test", novelty.compute())
