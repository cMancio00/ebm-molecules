from torch import nn
from typing import List, Any, Union
import torch
import numpy as np
from collections import deque


class SamplerWithBuffer(nn.Module):
    """
    Base class for all samplers
    """

    def __init__(self, max_len_buffer: int = 1000):
        super().__init__()
        self.max_len_buffer = max_len_buffer
        self.num_classes = None
        self.buffer = None
        self.register_buffer('_fake_p', torch.zeros(1))  # fake buffer just to get the device

    @property
    def device(self):
        return self.get_buffer('_fake_p').device

    def init_buffer(self):
        self.buffer = []
        for i in range(self.num_classes):
            self.buffer.append(self.generate_random_samples(self.max_len_buffer//10, collate=False))

    def get_negative_samples(self, model: nn.Module, labels: torch.Tensor,
                             steps: int = 60, step_size: float = 10.0) -> Any:
        """
        Function for getting a new batch of sampled tensors via MCMC.
        Inputs:
            steps - Number of iterations in the MCMC.
            step_size - Learning rate nu in the algorithm above
        """
        np_labels = labels.cpu().numpy()
        num_samples = labels.shape[0]
        mcmc_starting_tensors = []
        mcmc_random_tensors = self.generate_random_samples(num_samples, collate=False)
        idx_rand = 0
        for i in range(num_samples):
            s = np.random.binomial(n=1, p=1-0.05)
            if s == 1:
                # take from buffer
                idx = np.random.randint(0, len(self.buffer[np_labels[i]]))
                mcmc_starting_tensors.append(self.buffer[np_labels[i]][idx])
            else:
                # generate from random samples
                mcmc_starting_tensors.append(mcmc_random_tensors[idx_rand])
                idx_rand += 1

        # Perform MCMC sampling
        mcmc_starting_tensors = self.collate_fn(mcmc_starting_tensors)
        mcmc_samples = self.generate_samples(model, labels, mcmc_starting_tensors, steps=steps, step_size=step_size)

        # save in the buffer
        for i in range(num_samples):
            self.buffer[np_labels[i]].insert(0, mcmc_samples[i:i+1])

        for i in range(len(self.buffer)):
            self.buffer[i] = self.buffer[i][:self.max_len_buffer]

        return mcmc_samples

    @staticmethod
    def generate_samples(model: nn.Module, labels: torch.Tensor,  start_point: Any = None, steps: int = 60,
                         step_size: float = 1.0) -> Any:
        """
        Function for generating new tensors via MCMC, given a model for :math:`E_{\\theta}`
        The MCMC algorith perform the following update:

        .. math::
            x_{k+1} = x_{k} - \\varepsilon \\nabla_{x} E_{\\theta} + \\omega

        :param model: Neural network to use for modeling :math:`E_{\theta}`
        :param start_point: Batch of PyG Data to start from for sampling
        :param labels: Labels for conditional generation
        :param steps: Number of iterations in the MCMC algorithm.
        :param step_size: Learning rate :math:`\varepsilon`
        :return: Sampled Batch from the Energy distribution
        """
        raise NotImplementedError()

    def generate_random_samples(self, num_samples: int, collate=True) -> Any:
        raise NotImplementedError()

    def collate_fn(self, data_list: List[Any]) -> Any:
        raise NotImplementedError()
