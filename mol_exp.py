import torch
from ebm import DeepEnergyModel
from data_modules import SBMDataModule
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.cli import LightningCLI

torch.set_float32_matmul_precision('high')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# TODO: this can be generalised for all graphs datasets
# To generalize remove data argument and pass it to the cli
# --data=SBMDataModule
# class must be added to __init_.py data_module
# or add it to the config.yaml
class MolLightningCLI(LightningCLI):

    def __init__(self, *args, **kwargs):
        super().__init__(model_class=DeepEnergyModel, *args, **kwargs)

    def after_instantiate_classes(self) -> None:
        self.model.sampler.num_classes = self.datamodule.num_classes
        self.model.sampler.num_node_features = self.datamodule.num_node_features
        self.model.sampler.num_edge_features = self.datamodule.num_edge_features


if __name__ == '__main__':

    cli = MolLightningCLI(
        seed_everything_default=42,
        trainer_defaults={
            'callbacks': [
                ModelCheckpoint(save_top_k=2,save_last=True,auto_insert_metric_name=True, mode="max",
                                monitor="accuracy/validation"),
                LearningRateMonitor("epoch")
            ]
        }

    )
