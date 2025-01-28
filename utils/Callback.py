import random
import torch
import lightning as pl
from lightning import Trainer, LightningModule
import torchvision
from utils.Sampler import Sampler

class GenerateCallback(pl.Callback):

    def __init__(self, num_steps: int=256, vis_steps: int=8, every_n_epochs: int=5):
        """Uses MCMC to sample tensors from the model and logs them in the Tensorboard at the end
        of training epochs.

        Args:
            num_steps (int, optional): Number of MCMC steps to take during generation. Defaults to 256.
            vis_steps (int, optional): Steps within generation to visualize. Defaults to 8.
            every_n_epochs (int, optional): When we want to generate tensors. Defaults to 5.
        
        For example: The default number of steps in MCMC is 256, if we set `vis_steps` to 8, we will
        visualize 1 image each 32 steps (256/8).
        """
        super().__init__()
        self.vis_steps = vis_steps
        self.num_steps = num_steps
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """Called on train epoch end. Generates tensors from the model and save them in the Tensorboard

        Args:
            trainer (Trainer): Trainer to use
            pl_module (LightningModule): Model to use
        """
        if trainer.current_epoch % self.every_n_epochs == 0:
            imgs = self.generate_imgs(pl_module)
            step_size = self.num_steps // self.vis_steps
            imgs_to_plot = imgs[step_size-1::step_size,0]
            grid = torchvision.utils.make_grid(imgs_to_plot, nrow=imgs_to_plot.shape[0], normalize=True)
            trainer.logger.experiment.add_image(f"Generation during Training", grid, global_step=trainer.current_epoch)

    def generate_imgs(self, pl_module: LightningModule):
        pl_module.eval()
        start_imgs = torch.rand((1,) + tuple(pl_module.hparams["img_shape"])).to(pl_module.device)
        start_imgs = start_imgs * 2 - 1
        torch.set_grad_enabled(True)
        imgs_per_step = Sampler.generate_samples(pl_module.cnn, start_imgs, steps=self.num_steps, step_size=10, return_tensors_each_step=True)
        torch.set_grad_enabled(False)
        pl_module.train()
        return imgs_per_step
    
class SamplerCallback(pl.Callback):

    def __init__(self, num_samples=32, every_n_epochs=5):
        """Samples from the MCMC buffer and save the tensors to the Tensorboard

        Args:
            num_samples (int, optional): Number of samples. Defaults to 32.
            every_n_epochs (int, optional): When we want to generate tensors. Defaults to 5.
        """
        super().__init__()
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """Called on taining epoch end. Samples from the MCMC buffer and saves the tensors to the Tensorboard

        Args:
            trainer (Trainer): _description_
            pl_module (LightningModule): _description_
        """
        if trainer.current_epoch % self.every_n_epochs == 0:
            tensors_from_mcmc_buffer = torch.cat(random.choices(pl_module.sampler.buffer, k=self.num_samples), dim=0)
            grid = torchvision.utils.make_grid(tensors_from_mcmc_buffer, nrow=4, normalize=True)
            trainer.logger.experiment.add_image("Samples from MCMC buffer", grid, global_step=trainer.current_epoch)

class OutlierCallback(pl.Callback):

    def __init__(self, batch_size=1024):
        super().__init__()
        self.batch_size = batch_size

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        with torch.no_grad():
            pl_module.eval()
            rand_imgs = torch.rand((self.batch_size,) + pl_module.hparams["img_shape"]).to(pl_module.device)
            rand_imgs = rand_imgs * 2 - 1.0
            rand_out = pl_module.cnn(rand_imgs).mean()
            pl_module.train()

        trainer.logger.experiment.add_scalar("rand_out", rand_out, global_step=trainer.current_epoch)
