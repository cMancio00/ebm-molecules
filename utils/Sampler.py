import itertools
import random
from typing import List

import torch
import numpy as np
from lightning import LightningModule
from torch_geometric.data import Data, Batch
from utils.graphs import generate_random_graph, concat_batches


class Sampler:

    def __init__(self, model: LightningModule, sample_size: int, max_len: int = 8192):
        """
        Inputs:
            model - Neural network to use for modeling E_theta
            img_shape - Shape of the images to model
            sample_size - Batch size of the samples
            max_len - Maximum number of data points to keep in the buffer
        """
        super().__init__()
        self.model = model
        # self.img_shape = tuple(img_shape)
        self.sample_size = sample_size
        self.max_len = max_len
        # self.buffer = [(torch.rand((1,) + img_shape) * 2 - 1) for _ in range(self.sample_size)]
        self.buffer: List[Data] = [generate_random_graph() for _ in range(self.sample_size)]

    def sample_new_tensor(self, labels, steps: int = 60, step_size: float = 10.0) -> Batch:
        """
        Function for getting a new batch of sampled tensors via MCMC.
        Inputs:
            steps - Number of iterations in the MCMC.
            step_size - Learning rate nu in the algorithm above
        """
        # Choose 95% of the batch from the buffer, 5% generate from scratch
        num_new_samples: int = np.random.binomial(self.sample_size, 0.05)
        print(f"Generating: {num_new_samples} new samples")
        if not num_new_samples == 0:
            new_rand_tensors: Batch = Batch.from_data_list([generate_random_graph() for _ in range(num_new_samples)])
        # Randomlly select the old tensor from the buffer
        # old_tensors = torch.cat(random.choices(self.buffer, k=self.sample_size - num_new_samples), dim=0)
        old_tensors: Batch = Batch.from_data_list(random.choices(self.buffer, k=self.sample_size - num_new_samples))
        # Concatenate the old tensors and the new ones in the batch dimension
        # mcmc_starting_tensors = torch.cat([new_rand_tensors, old_tensors], dim=0).detach().to(self.model.device)
        if not num_new_samples == 0:
            mcmc_starting_tensors: Batch = concat_batches([new_rand_tensors, old_tensors])
        else:
            mcmc_starting_tensors: Batch = old_tensors
        # Perform MCMC sampling
        # mcmc_samples = Sampler.generate_samples(self.model, mcmc_starting_tensors, labels, steps=steps, step_size=step_size)
        # simulate no mcmc for testing
        mcmc_samples: Batch = mcmc_starting_tensors

        # Add new images to the buffer and remove old ones if needed
        # self.buffer = list(mcmc_samples.to(torch.device("cpu")).chunk(self.sample_size, dim=0)) + self.buffer
        # self.buffer = self.buffer[:self.max_len]

        self.buffer = list(itertools.chain(Batch.to_data_list(mcmc_samples), self.buffer))
        self.buffer = self.buffer[:self.max_len]
        return mcmc_samples

    @staticmethod
    def generate_samples(model: LightningModule, mcmc_starting_tensors: torch.Tensor, lables, steps: int = 60, step_size: float = 10.0,
                         return_tensors_each_step: bool = False) -> torch.Tensor:
        """
        Function for generating new tensors via MCMC, given a model for :math:`E_{\\theta}`
        The MCMC algorith perform the following update:

        .. math::
            x_{k+1} = x_{k} - \\varepsilon \\nabla_{x} E_{\\theta} + \\omega

        :param model: Neural network to use for modeling :math:`E_{\theta}`
        :param mcmc_starting_tensors: Tensors to start from for sampling. If you want to generate new images, enter noise between -1 and 1.
        :param steps: Number of iterations in the MCMC algorithm.
        :param step_size: Learning rate :math:`\\varepsilon`
        :param return_tensors_each_step: If True, we return the sample at every iteration of the MCMC
        :return: Sampled Tensors from the Energy distribution
        """

        # Save the training state of the model ad activate only gradient for the input
        is_training = model.training
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        mcmc_starting_tensors.requires_grad = True
        # Enable gradient calculation if not already the case
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # We use a buffer tensor in which we generate noise each loop iteration.
        noise = torch.randn(mcmc_starting_tensors.shape, device=mcmc_starting_tensors.device)

        # List for storing generations at each step (for later analysis)
        tensors_each_step = []

        # MCMC
        for _ in range(steps):
            # Part 1: Add noise to the input.
            noise.normal_(0, 0.005)
            mcmc_starting_tensors.data.add_(noise.data)
            mcmc_starting_tensors.data.clamp_(min=-1.0, max=1.0)

            # Part 2: Get the energy and calculate the gradient of the input
            energy: torch.Tensor = -(model(mcmc_starting_tensors)[torch.arange(lables.size(0)),lables])
            # energy.logsumexp(dim=0)
            energy.sum().backward()
            mcmc_starting_tensors.grad.data.clamp_(-0.03, 0.03)

            # Apply gradients to our current samples
            mcmc_starting_tensors.data.add_(-step_size * mcmc_starting_tensors.grad.data)
            mcmc_starting_tensors.grad.detach_()
            mcmc_starting_tensors.grad.zero_()
            mcmc_starting_tensors.data.clamp_(min=-1.0, max=1.0)

            if return_tensors_each_step:
                tensors_each_step.append(mcmc_starting_tensors.clone().detach())

        # Reactivate gradients for parameters for training
        for p in model.parameters():
            p.requires_grad = True
        model.train(is_training)

        # Reset gradient calculation to setting before this function
        torch.set_grad_enabled(had_gradients_enabled)

        if return_tensors_each_step:
            return torch.stack(tensors_each_step, dim=0)
        else:
            return mcmc_starting_tensors


