import math
import random
from typing import Any

import matplotlib.pyplot as plt
import torch
import lightning as pl
from lightning import Trainer, LightningModule
import torchvision


def _plot_data(func_to_plot, data_list, num_rows, subplot_size=(2, 2)):
    n_samples = len(data_list)
    num_cols = n_samples // num_rows
    f, ax_list = plt.subplots(num_rows, num_cols, figsize=(subplot_size[0] * num_cols, subplot_size[1] * num_rows))
    ax_list = ax_list.reshape(-1)
    plt.setp(ax_list, xticks=[], yticks=[])

    for i, data in enumerate(data_list):
        func_to_plot(data, ax_list[i])

    return f

def _generate_grid_samples(trainer: "pl.Trainer", pl_module: "pl.LightningModule", n_plot_during_generation: int = 10) -> list[tuple[torch.tensor, torch.tensor]]:
    num_classes = pl_module.sampler.num_classes
    device = pl_module.device
    labels = torch.arange(num_classes, device=device)

    start_x, _ = pl_module.sampler.generate_random_batch(batch_size=labels.shape[0], device=device,
                                                         collate=True)
    all_sample = [start_x.clone().cpu()]

    n_steps = pl_module.hparams.mcmc_steps_gen

    mcmc_steps = n_steps // n_plot_during_generation

    for _ in range(n_plot_during_generation):
        start_x = pl_module.sampler.MCMC_generation(model=pl_module.nn_model,
                                                    steps=mcmc_steps,
                                                    step_size=pl_module.hparams.mcmc_learning_rate_gen,
                                                    labels=labels,
                                                    starting_x=start_x)
        all_sample.append(start_x.clone().cpu())

    n_cols = len(all_sample)
    data_list = [(None, None) for _ in range(n_cols * num_classes)]
    for j, s in enumerate(all_sample):
        for i in range(num_classes):
            data_list[i * n_cols + j] = (s[i], torch.tensor(i, device='cpu'))
    return data_list


class GenerateCallback(pl.Callback):

    def __init__(self, n_plot_during_generation: int = 10, every_n_epochs: int = 5, n_replica_during_test: int = 5):
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
        self.n_replica_during_test = n_replica_during_test

    #def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called on train epoch end. Generates tensors from the model and save them in the Tensorboard

        Args:
            trainer (Trainer): Trainer to use
            pl_module (LightningModule): Model to use
        """
        if trainer.current_epoch % self.every_n_epochs == 0 and trainer.state.stage != 'sanity_check':
            data_list = _generate_grid_samples(trainer, pl_module, self.n_plot_during_generation)

            f = _plot_data(pl_module.sampler.plot_sample, data_list, pl_module.sampler.num_classes)
            trainer.logger.experiment.add_figure(f"Generation during Training", f, global_step=trainer.current_epoch)

    @torch.inference_mode(False)
    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        for i in range(self.n_replica_during_test):
            data_list = _generate_grid_samples(trainer, pl_module, self.n_plot_during_generation)

            f = _plot_data(pl_module.sampler.plot_sample, data_list, pl_module.sampler.num_classes)
            trainer.logger.experiment.add_figure(f"Best Checkpoint Generation Test", f, global_step=i)


class BufferSamplerCallback(pl.Callback):

    def __init__(self, num_samples=16, every_n_epochs=5):
        """Samples from the MCMC buffer and save the tensors to the Tensorboard

        Args:
            num_samples (int, optional): Number of samples. Defaults to 64.
            every_n_epochs (int, optional): When we want to generate tensors. Defaults to 5.
        """
        super().__init__()
        self.num_samples = num_samples
        self.num_rows = int(math.sqrt(num_samples))
        self.every_n_epochs = every_n_epochs

    #def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called on training epoch end. Samples from the MCMC buffer and saves the tensors to the Tensorboard

        Args:
            trainer (Trainer): _description_
            pl_module (LightningModule): _description_
        """
        if trainer.current_epoch % self.every_n_epochs == 0 and trainer.state.stage != 'sanity_check':
            idx_to_plot = random.sample(range(len(pl_module.sampler.buffer)), self.num_samples)

            data_list = [(pl_module.sampler.buffer[i][0].clone().cpu(), pl_module.sampler.buffer[i][1].clone().cpu())
                         for i in idx_to_plot]
            f = _plot_data(pl_module.sampler.plot_sample, data_list, self.num_rows)

            trainer.logger.experiment.add_figure("Samples from MCMC buffer", f, global_step=trainer.current_epoch)


class PlotBatchCallback(pl.Callback):
    def __init__(self, num_samples=16, every_n_epochs=5):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.num_rows = int(math.sqrt(num_samples))
        self.num_cols = self.num_rows
        self.num_samples = self.num_rows * self.num_cols

    #def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    #                         ) -> None:
    def on_validation_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:

        if (trainer.current_epoch % self.every_n_epochs == 0 and batch_idx == 0 and
                trainer.state.stage != 'sanity_check'):
            x, y = batch
            x, y = x.clone().cpu(), y.clone().cpu()
            idx_to_plot = random.sample(range(len(x)), self.num_samples)
            data_list = [(x[i], y[i]) for i in idx_to_plot]
            f = _plot_data(pl_module.sampler.plot_sample, data_list, self.num_rows)

            trainer.logger.experiment.add_figure("Batch samples", f, global_step=trainer.current_epoch)


class ChangeClassCallback(pl.Callback):
    def __init__(
            self,
            n_plot_during_generation: int = 10,
            samples_to_plot: int = 10,
            batch_to_change: int = 1,
            mcmc_steps: None | int = None):

        super().__init__()
        self.n_plot_during_generation = n_plot_during_generation
        self.samples_to_plot = samples_to_plot
        self.batch_to_change = batch_to_change
        self.mcmc_steps = mcmc_steps

    @torch.inference_mode(False)
    def on_test_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:

        if self.mcmc_steps is None:
            n_steps = pl_module.hparams.mcmc_steps_gen
        else:
            n_steps = self.mcmc_steps

        if batch_idx in range(self.batch_to_change):
            num_classes = pl_module.sampler.num_classes

            start_x, labels = batch
            start_x = start_x[:self.samples_to_plot].clone()
            labels = labels[:self.samples_to_plot].clone()

            labels = (labels + 1) % num_classes

            all_sample = [start_x.clone().cpu()]

            # n_steps = pl_module.hparams.mcmc_steps_gen

            mcmc_steps = n_steps // self.n_plot_during_generation

            for _ in range(self.n_plot_during_generation):
                start_x = pl_module.sampler.MCMC_generation(model=pl_module.nn_model,
                                                            steps=mcmc_steps,
                                                            step_size=pl_module.hparams.mcmc_learning_rate_gen,
                                                            labels=labels,
                                                            starting_x=start_x)
                all_sample.append(start_x.clone().cpu())

            n_cols = len(all_sample)
            labels = labels.cpu()
            data_list = [(None, None) for _ in range(n_cols * self.samples_to_plot)]
            for j, s in enumerate(all_sample):
                for i in range(self.samples_to_plot):
                    data_list[i * n_cols + j] = (s[i], labels[i])


            f = _plot_data(pl_module.sampler.plot_sample, data_list, self.samples_to_plot)
            trainer.logger.experiment.add_figure(f"Change Class/test", f, global_step=batch_idx)


