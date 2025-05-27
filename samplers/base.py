from torch import nn
from typing import List, Any, Tuple, Union
import torch
import numpy as np
import random
import matplotlib.pyplot as plt


class SamplerWithBuffer(nn.Module):
    """
    Base class for all samplers
    """

    def __init__(self, max_len_buffer: int):
        super().__init__()
        self.max_len_buffer = max_len_buffer
        self.num_classes = None
        self.buffer = []
        self.register_buffer('_fake_p', torch.zeros(1))  # fake buffer just to get the device

    @property
    def device(self):
        return self.get_buffer('_fake_p').device

    def init_buffer(self):
        self.buffer = []

    def get_negative_batch(self, model: nn.Module, batch_size: int,
                           steps: int, step_size: float) -> Tuple[Any, torch.Tensor]:
        """
        Function for getting a new batch of sampled tensors via MCMC and buffer.
        Inputs:
            steps - Number of iterations in the MCMC.
            step_size - Learning rate nu in the algorithm above
        """
        n_to_sample = min(np.random.binomial(n=batch_size, p=1 - 0.05), len(self.buffer))
        sampled_indexes: List[int] = random.sample(range(len(self.buffer)), n_to_sample)

        random_elements: List[Tuple[Any, torch.Tensor]] = self.generate_random_batch(batch_size-len(sampled_indexes),
                                                                                     collate=False)

        neg_x, neg_y = self.collate_fn(random_elements + [self.buffer[i] for i in sampled_indexes])

        mcmc_x = self.MCMC_generation(model, steps=steps, step_size=step_size, labels=neg_y, starting_x=neg_x)

        # save in the buffer
        for i in range(batch_size):
            self.buffer.append((mcmc_x[i], neg_y[i]))

        if len(self.buffer) > self.max_len_buffer:
            # take the last ones
            self.buffer = self.buffer[-self.max_len_buffer:]

        return mcmc_x, neg_y

    def MCMC_generation(self, model: nn.Module, steps: int, step_size: float, labels: torch.Tensor,
                        starting_x: Any = None) -> Any:
        """
        Function for generating new tensors via MCMC, given a model for :math:`E_{\\theta}`
        The MCMC algorith perform the following update:

        .. math::
            x_{k+1} = x_{k} - \\varepsilon \\nabla_{x} E_{\\theta} + \\omega

        :param model: Neural network to use for modeling :math:`E_{\theta}`
        :param starting_x: Batch of PyG Data to start from for sampling
        :param labels: Labels for conditional generation
        :param steps: Number of iterations in the MCMC algorithm.
        :param step_size: Learning rate :math:`\varepsilon`
        :return: Sampled Batch from the Energy distribution
        """
        # save model state
        is_training = model.training
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)


        batch_size = labels.shape[0]
        device = labels.device

        if starting_x is None:
            starting_x, _ = self.generate_random_batch(batch_size, device, collate=True)

        generated_batch = self._MCMC_generation(model, steps, step_size, labels, starting_x)

        # Reactivate gradients for parameters for training
        for p in model.parameters():
            p.requires_grad = True
        model.train(is_training)

        # Reset gradient calculation to setting before this function
        torch.set_grad_enabled(had_gradients_enabled)

        return generated_batch

    def _MCMC_generation(self, model: nn.Module, steps: int, step_size: float, labels: torch.Tensor,
                         starting_x: Any) -> Tuple[Any, torch.Tensor]:
        """
        Function for generating new tensors via MCMC, given a model for :math:`E_{\\theta}`
        The MCMC algorith perform the following update:

        .. math::
            x_{k+1} = x_{k} - \\varepsilon \\nabla_{x} E_{\\theta} + \\omega

        :param model: Neural network to use for modeling :math:`E_{\theta}`
        :param steps: Number of iterations in the MCMC algorithm.
        :param step_size: Learning rate :math:`\varepsilon`
        :param labels: Labels for conditional generation
        :param starting_x: Batch of PyG Data to start from for sampling
        :return: Sampled Batch from the Energy distribution
        """
        raise NotImplementedError()

    @torch.no_grad()
    def generate_random_batch(self, batch_size: int, device=None, collate: bool = True) -> (
            Union[List[Tuple[Any, torch.Tensor]], Tuple[Any, torch.Tensor]]):
        raise NotImplementedError()

    @staticmethod
    def collate_fn(data_list: List[Tuple[Any, torch.Tensor]]) -> Tuple[Any, torch.Tensor]:
        raise NotImplementedError()

    def plot_sample(self, s: Tuple[Any, torch.Tensor], ax: plt.Axes) -> None:
        ax.imshow(s[0])
        ax.set_title(f'Label {s[1]}')
