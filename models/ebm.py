import torch

from models import Small_CNN
from utils.Sampler import Sampler
import lightning as pl
import torch.optim as optim

class DeepEnergyModel(pl.LightningModule):

    def __init__(self, img_shape : tuple[int, int, int] = (1, 28, 28), batch_size : int = 32, alpha=0.1, lr=1e-4, beta1=0.0, mcmc_steps: int = 60, mcmc_learning_rate: float = 10.0,**CNN_args):
        super().__init__()
        self.save_hyperparameters()
        self.cnn = Small_CNN(**CNN_args)
        self.batch_size = batch_size
        self.sampler = Sampler(self.cnn, img_shape=tuple(img_shape), sample_size=self.batch_size)
        self.mcmc_steps = mcmc_steps
        self.mcmc_learning_rate = mcmc_learning_rate

    def forward(self, x):
        z = self.cnn(x)
        return z

    def configure_optimizers(self):
        # Energy models can have issues with momentum as the loss surfaces changes with its parameters.
        # Hence, we set it to 0 by default.
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.97) # Exponential decay over epochs
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # We add minimal noise to the original images to prevent the model from focusing on purely "clean" inputs
        real_imgs, _ = batch
        small_noise = torch.randn_like(real_imgs) * 0.005
        real_imgs.add_(small_noise).clamp_(min=-1.0, max=1.0)

        # Sample fake tensors (samples n = batch_size tensors)
        fake_imgs = self.sampler.sample_new_tensor(steps=self.mcmc_steps, step_size=self.mcmc_learning_rate)

        # Predict energy score for all images
        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim=0)

        # Calculate losses
        # reg_loss = self.hparams.alpha * (real_out ** 2 + fake_out ** 2).mean()
        reg_loss = torch.pow(real_imgs - fake_imgs, 2).sum()
        cdiv_loss = fake_out.mean() - real_out.mean()
        loss = cdiv_loss + reg_loss
        
        # Logging
        self.log('loss', loss)
        self.log('loss_regularization', reg_loss)
        self.log('loss_contrastive_divergence', cdiv_loss)
        self.log('metrics_avg_real', real_out.mean())
        self.log('metrics_avg_fake', fake_out.mean())
        return loss
    
    def validation_step(self, batch, batch_idx):
        # For validating, we calculate the contrastive divergence between purely random images and unseen examples
        # Note that the validation/test step of energy-based models depends on what we are interested in the model
        real_imgs, _ = batch
        fake_imgs = torch.rand_like(real_imgs) * 2 - 1

        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim=0)

        cdiv = fake_out.mean() - real_out.mean()
        self.log('val_contrastive_divergence', cdiv)
        self.log('val_fake_out', fake_out.mean())
        self.log('val_real_out', real_out.mean())

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

        