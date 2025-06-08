import lightning as pl
from lightning import Trainer, LightningModule
import torch
from torch.nn.utils.parametrizations import spectral_norm
from torch.nn.utils.parametrize import is_parametrized
from torch import Tensor
from torchmetrics.functional import accuracy

class SpectralNormalizationCallback(pl.Callback):

    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        for module in pl_module.nn_model.modules():
            if hasattr(module, "weight") and ("weight" in dict(module.named_parameters())):
                if not is_parametrized(module, "weight"):
                    spectral_norm(module, name="weight", n_power_iterations=1)

class SelfAccuracyCallback(pl.Callback):

    def __init__(self, batch_size: int = 256):
        super().__init__()
        self.batch_size = batch_size

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:

        samples, labels = pl_module.sampler.get_negative_batch(model=pl_module.nn_model, batch_size=self.batch_size,
                                                                  steps=pl_module.hparams.mcmc_steps_gen,
                                                                  step_size=pl_module.hparams.mcmc_learning_rate_gen)

        device = labels.device
        # idx = torch.arange(self.batch_size, device=device)
        energy: Tensor = pl_module.nn_model(samples)
        pred = energy.argmax(dim=-1)

        pl_module.log("SelfAccuracy/test",
                 accuracy(pred, labels, task='multiclass',num_classes=energy.shape[-1]),
                 batch_size=self.batch_size,
                 on_step=False, on_epoch=True)

