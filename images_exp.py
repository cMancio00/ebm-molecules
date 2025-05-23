import torch
from ebm import DeepEnergyModel
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.cli import LightningCLI

torch.set_float32_matmul_precision('high')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# TODO: this can be generalised for all images datasets
class ImagesLightningCLI(LightningCLI):

    def __init__(self, *args, **kwargs):
        super().__init__(model_class=DeepEnergyModel, *args, **kwargs)

    def after_instantiate_classes(self) -> None:
        self.model.sampler.num_classes = self.datamodule.num_classes
        self.model.sampler.img_shape = self.datamodule.img_shape


if __name__ == '__main__':

    cli = ImagesLightningCLI(
        seed_everything_default=42,
        trainer_defaults={
            'callbacks': [
                ModelCheckpoint(save_top_k=1,auto_insert_metric_name=True,
                                monitor="accuracy/validation"),
                LearningRateMonitor("epoch")
            ]
        }

    )
