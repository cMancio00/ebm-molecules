import random
from typing import List
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import torch
from torch import Tensor
import lightning as pl
from lightning import Trainer, LightningModule
import torchvision
from torch.nn.utils.parametrizations import spectral_norm
from torch.nn.utils.parametrize import is_parametrized
from utils.Sampler import Sampler
from utils.graphs import superpixels_to_image
import numpy as np
from torchvision.utils import make_grid

class GenerateCallback(pl.Callback):

    def __init__(self, num_steps: int=256, vis_steps: int=10, every_n_epochs: int=5, tensors_to_generate : int = 10):
        """Uses MCMC to sample tensors from the model and logs them in the Tensorboard at the end
        of training epochs.

        Args:
            num_steps (int, optional): Number of MCMC steps to take during generation. Defaults to 256.
            vis_steps (int, optional): Steps within generation to visualize. Defaults to 8.
            every_n_epochs (int, optional): When we want to generate tensors. Defaults to 5.
            tensors_to_generate (int, optional): Number of tensors to generate. Defaults to 1
        
        For example: The default number of steps in MCMC is 256, if we set `vis_steps` to 8, we will
        visualize 1 image each 32 steps (256/8).
        """
        super().__init__()
        self.vis_steps = vis_steps
        self.num_steps = num_steps
        self.every_n_epochs = every_n_epochs
        self.tensors_to_generate = tensors_to_generate

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """Called on train epoch end. Generates tensors from the model and save them in the Tensorboard

        Args:
            trainer (Trainer): Trainer to use
            pl_module (LightningModule): Model to use
        """
        if trainer.current_epoch % self.every_n_epochs == 0:
            imgs = self.generate_imgs(pl_module)
            step_size = self.num_steps // self.vis_steps
            imgs_to_plot = imgs[step_size-1::step_size]
            grid = torchvision.utils.make_grid(imgs_to_plot.reshape(-1, 1, 28, 28), nrow=imgs_to_plot.shape[0], normalize=True)
            trainer.logger.experiment.add_image(f"Generation during Training", grid, global_step=trainer.current_epoch)

    def generate_imgs(self, pl_module: LightningModule):
        pl_module.eval()
        start_imgs = torch.rand((self.tensors_to_generate,) + tuple(pl_module.hparams["img_shape"])).to(pl_module.device)
        start_imgs = start_imgs * 2 - 1
        torch.set_grad_enabled(True)
        labels : Tensor = torch.arange(10) #torch.randint(0,10,(10,))
        imgs_per_step = Sampler.generate_samples(pl_module.cnn, start_imgs, lables=labels ,steps=self.num_steps, step_size=10, return_tensors_each_step=True)
        torch.set_grad_enabled(False)
        pl_module.train()
        return imgs_per_step
    
class BufferSamplerCallback(pl.Callback):

    def __init__(self, num_samples=64, num_rows=4, every_n_epochs=5):
        """Samples from the MCMC buffer and save the tensors to the Tensorboard

        Args:
            num_samples (int, optional): Number of samples. Defaults to 64.
            every_n_epochs (int, optional): When we want to generate tensors. Defaults to 5.
        """
        super().__init__()
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs
        self.num_rows = num_rows

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """Called on training epoch end. Samples from the MCMC buffer and saves the tensors to the Tensorboard

        Args:
            trainer (Trainer): _description_
            pl_module (LightningModule): _description_
        """
        if trainer.current_epoch % self.every_n_epochs == 0:
            sampled_indexes: List[int] = random.choices(
                range(len(pl_module.sampler.buffer)),
                k=self.num_samples
            )
            batch = pl_module.sampler.buffer[sampled_indexes]

            col = (len(batch) // self.num_rows)
            images: List[Tensor] = []
            for i in range(len(batch[:(col * self.num_rows)])):
                image = superpixels_to_image(batch[i])
                images.append(
                    torch.from_numpy(image).permute(2, 1, 0)
                )
            grid = make_grid(images, nrow=self.num_rows)

            trainer.logger.experiment.add_image("Samples from MCMC buffer", grid, global_step=trainer.current_epoch)


class SpectralNormalizationCallback(pl.Callback):

    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        print("Spectral Normalization Added")
        for module in pl_module.cnn.modules():
            if hasattr(module, "weight") and ("weight" in dict(module.named_parameters())):
                if not is_parametrized(module, "weight"):
                    spectral_norm(module, name="weight", n_power_iterations=1)
