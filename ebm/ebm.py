import torch
from pytorch_lightning.utilities import grad_norm
from torch import Tensor
from torchmetrics.functional import accuracy
import torch.nn.functional as F
import lightning as pl
import torch.optim as optim
from samplers.base import SamplerWithBuffer
from torch import nn


class DeepEnergyModel(pl.LightningModule):

    def __init__(self, nn_model: nn.Module, sampler: SamplerWithBuffer, optimizer_type: str = 'sgd',
                 mcmc_steps_tr: int = 10, mcmc_learning_rate_tr: float = 1.0, # hparams for the mcmc sampling during training
                 mcmc_steps_gen: int = 10, mcmc_learning_rate_gen: float = 1.0,  # hparams for the mcmc sampling during validation
                 alpha_penalty=0.1, alpha_ce=1, alpha_cd=1, lr=1e-4, beta1=0.0, # hparams for the optimizer
                 lr_step_size: int = 1, gamma: float = 0.97):  # hparams for the LR_scheduler
        super().__init__()
        self.save_hyperparameters()
        self.nn_model = nn_model
        self.sampler = sampler
        self.optimizer_type = optimizer_type

    def configure_optimizers(self):
        if self.optimizer_type not in ["sgd", "adam"]:
            raise ValueError(f'Optimizer must be "sgd" or "adam"')
        if self.optimizer_type == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr)
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.lr_step_size, gamma=self.hparams.gamma)
        return [optimizer] , [scheduler]

    def training_step(self, batch, batch_idx):
        x, labels = batch
        batch_size = labels.size(0)
        device = labels.device
        idx = torch.arange(batch_size, device=device)

        # positive phase
        positive_energy: Tensor = self.nn_model(x)
        pred = positive_energy.argmax(dim=-1)
        cross_entropy: Tensor = F.cross_entropy(positive_energy, labels)
        positive_energy = positive_energy[idx, labels]

        # negative phase
        neg_samples, neg_labels = self.sampler.get_negative_batch(model=self.nn_model, batch_size=batch_size,
                                                                  steps=self.hparams.mcmc_steps_tr,
                                                                  step_size=self.hparams.mcmc_learning_rate_tr)
        negative_energy: Tensor = self.nn_model(neg_samples)[idx, neg_labels]

        cd_generative_loss: Tensor = (negative_energy - positive_energy).mean()
        penalty = (positive_energy ** 2 + negative_energy ** 2).mean()

        loss: Tensor = self.hparams.alpha_cd * cd_generative_loss + self.hparams.alpha_ce * cross_entropy + self.hparams.alpha_penalty * penalty

        # log the values
        self.log('loss/total', loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('loss/contrastive_divergence', cd_generative_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('loss/penalty', penalty, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("loss/cross_entropy", cross_entropy, on_step=False, on_epoch=True, batch_size=batch_size)

        self.log('energy/positive_phase', positive_energy.mean(), on_step=False, on_epoch=True, batch_size=batch_size)
        self.log('energy/negative_phase', negative_energy.mean(), on_step=False, on_epoch=True, batch_size=batch_size)

        self.log("accuracy/training",
                 accuracy(pred, labels, task='multiclass',num_classes=positive_energy.shape[-1]),
                 batch_size=batch_size,
                 on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        positive_energy: torch.Tensor = self.nn_model(x)
        batch_size, num_classes = positive_energy.shape

        #generated_x = self.sampler.MCMC_generation(self.nn_model, self.hparams.mcmc_steps_gen,
        #                                           self.hparams.mcmc_learning_rate_gen, labels)
        #negative_energy: Tensor = self.nn_model(generated_x)

        cross_entropy: Tensor = F.cross_entropy(positive_energy, labels)
        pred = positive_energy.argmax(dim=-1)

        #positive_energy = positive_energy[torch.arange(labels.size(0)), labels]
        #negative_energy = negative_energy[torch.arange(labels.size(0)), labels]
        #loss: Tensor = (negative_energy - positive_energy).mean()

        #self.log('val_contrastive_divergence', loss, batch_size=batch_size, on_step=False, on_epoch=True)
        #self.log("Cross Entropy Validation", cross_entropy, batch_size=batch_size, on_step=False, on_epoch=True)
        self.log("accuracy/validation", accuracy(pred, labels, task='multiclass', num_classes=num_classes),
                 batch_size=batch_size, on_step=False, on_epoch=True)

        return cross_entropy

    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint["state_dict"]
        new_state_dict = {}
        for key, value in state_dict.items():
            if "parametrizations.weight.original" in key:
                new_key = key.replace("parametrizations.weight.original", "weight")
                new_state_dict[new_key] = value
            elif "parametrizations.weight.0._u" in key or "parametrizations.weight.0._v" in key:
                continue
            else:
                new_state_dict[key] = value
        checkpoint["state_dict"] = new_state_dict

    def on_train_start(self) -> None:
        self.sampler.init_buffer()

    def on_train_batch_end (self, outputs, batch, batch_idx) -> None:
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.nn_model, norm_type=2)
        self.log_dict(norms)

