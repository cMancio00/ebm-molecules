import math
import random
import torch
import lightning as pl
from lightning import Trainer, LightningModule
import torchvision
from torch.nn.utils.parametrizations import spectral_norm
from torch.nn.utils.parametrize import is_parametrized

from torchvision.utils import make_grid

class GenerateCallback(pl.Callback):

    def __init__(self, n_plot_during_generation: int=10, every_n_epochs: int=5):
        """Uses MCMC to sample tensors from the model and logs them in the Tensorboard at the end
        of training epochs.

        Args:
            num_steps (int, optional): Number of MCMC steps to take during generation. Defaults to 256.
            n_plot_during_generation (int, optional): Number of samples to visualize during the generation. Defaults to 10.
            every_n_epochs (int, optional): When we want to generate tensors. Defaults to 5.
            tensors_to_generate (int, optional): Number of tensors to generate. Defaults to 1
        
        For example: The default number of steps in MCMC is 256, if we set `vis_steps` to 8, we will
        visualize 1 image each 32 steps (256/8).
        """
        super().__init__()
        self.n_plot_during_generation = n_plot_during_generation
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """Called on train epoch end. Generates tensors from the model and save them in the Tensorboard

        Args:
            trainer (Trainer): Trainer to use
            pl_module (LightningModule): Model to use
        """
        if trainer.current_epoch % self.every_n_epochs == 0:
            num_classes = pl_module.sampler.num_classes
            device = pl_module.device
            labels = torch.arange(num_classes, device=device)

            start_x, _ = pl_module.sampler.generate_random_batch(batch_size=labels.shape[0], device=device, collate=True)
            all_sample = [pl_module.sampler.plot_sample(start_x).clone()]

            n_steps = pl_module.hparams.mcmc_steps_gen

            mcmc_steps = n_steps // self.n_plot_during_generation

            for _ in range(self.n_plot_during_generation):

                start_x = pl_module.sampler.MCMC_generation(model=pl_module.nn_model,
                                                            steps=mcmc_steps,
                                                            step_size=pl_module.hparams.mcmc_learning_rate_gen,
                                                            labels=labels,
                                                            starting_x=start_x)
                all_sample.append(pl_module.sampler.plot_sample(start_x).clone())

            all_images = torch.stack(all_sample, dim=1).view((-1,) + all_sample[0].shape[1:])
            grid = torchvision.utils.make_grid(all_images, nrow=self.n_plot_during_generation+1)
            trainer.logger.experiment.add_image(f"Generation during Training", grid, global_step=trainer.current_epoch)


class BufferSamplerCallback(pl.Callback):

    def __init__(self, num_samples=64, every_n_epochs=5):
        """Samples from the MCMC buffer and save the tensors to the Tensorboard

        Args:
            num_samples (int, optional): Number of samples. Defaults to 64.
            every_n_epochs (int, optional): When we want to generate tensors. Defaults to 5.
        """
        super().__init__()
        self.num_samples = num_samples
        self.num_rows = int(math.sqrt(num_samples))
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        """Called on training epoch end. Samples from the MCMC buffer and saves the tensors to the Tensorboard

        Args:
            trainer (Trainer): _description_
            pl_module (LightningModule): _description_
        """
        if trainer.current_epoch % self.every_n_epochs == 0:
            idx_to_plot = random.sample(range(len(pl_module.sampler.buffer)), self.num_samples)
            imgs_to_plot = [pl_module.sampler.plot_sample(pl_module.sampler.buffer[i][0]) for i in idx_to_plot]

            grid = make_grid(imgs_to_plot, nrow=self.num_rows)

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
