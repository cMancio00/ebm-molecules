import random
from typing import List
import torch
import numpy as np
from lightning import LightningModule
from torch_geometric.data import Batch
from utils.graphs import generate_random_graph
from utils.data import DenseData, densify_data


class Sampler:

    def __init__(self, model: LightningModule, sample_size: int, max_len: int = 8192):
        """
        Inputs:
            model - Neural network to use for modeling E_theta
            sample_size - Batch size of the samples
            max_len - Maximum number of data points to keep in the buffer
        """
        super().__init__()
        self.model: LightningModule = model
        self.sample_size: int = sample_size
        self.max_len: int = max_len
        self.buffer = None

    def init_buffer(self):
        self.buffer: DenseData = densify_data(
            Batch.from_data_list(
                [generate_random_graph(device=self.model.device) for _ in range(self.sample_size)]
            )
        )

    def sample_new_tensor(self, labels: torch.Tensor, steps: int = 60, step_size: float = 10.0) -> DenseData:
        """
        Function for getting a new batch of sampled tensors via MCMC.
        Inputs:
            steps - Number of iterations in the MCMC.
            step_size - Learning rate nu in the algorithm above
        """
        sampled_indexes: List[int] = random.sample(
            range(len(self.buffer)),
            np.random.binomial(self.sample_size, 1 - 0.05)
        )
        old_tensors: DenseData = self.buffer[sampled_indexes]

        if not len(sampled_indexes) == self.sample_size:
            new_rand_tensors: DenseData = densify_data(
                Batch.from_data_list(
                    [generate_random_graph(device=self.model.device) for _ in range(self.sample_size - len(sampled_indexes))]
                )
            )
            mcmc_starting_tensors: DenseData = new_rand_tensors + old_tensors

        else:
            mcmc_starting_tensors: DenseData = old_tensors

        # Perform MCMC sampling
        mcmc_samples = Sampler.generate_samples(self.model, mcmc_starting_tensors, labels, steps=steps, step_size=step_size)
        # mcmc_samples: DenseData = mcmc_starting_tensors
        self.buffer = (mcmc_samples + self.buffer)[:self.max_len]
        return mcmc_samples

    @staticmethod
    def generate_samples(model: LightningModule, batch: DenseData, labels: torch.Tensor,
                         steps: int = 60, step_size: float = 1.0) -> DenseData:
        """
        Function for generating new tensors via MCMC, given a model for :math:`E_{\\theta}`
        The MCMC algorith perform the following update:

        .. math::
            x_{k+1} = x_{k} - \\varepsilon \\nabla_{x} E_{\\theta} + \\omega

        :param model: Neural network to use for modeling :math:`E_{\theta}`
        :param batch: Batch of PyG Data to start from for sampling
        :param labels: Labels for conditional generation
        :param steps: Number of iterations in the MCMC algorithm.
        :param step_size: Learning rate :math:`\varepsilon`
        :return: Sampled Batch from the Energy distribution
        """

        # Save the training state of the model ad activate only gradient for the input
        is_training = model.training
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        had_gradients_enabled = torch.is_grad_enabled()

        x, adj, mask = batch.x, batch.adj, batch.mask
        x.requires_grad_()
        adj.requires_grad_()

        noise_x = torch.randn(x.shape, device=model.device)
        noise_adj = torch.randn(adj.shape, device=model.device)

        # MCMC
        # batch.requires_grad = True
        for i in range(steps):
            noise_x.normal_(0, 0.005)
            noise_adj.normal_(0, 0.005)
            x.retain_grad()
            adj.retain_grad()
            energy = -model(x, adj, mask)[torch.arange(labels.size(0)), labels]
            energy.sum().backward(retain_graph=True)
            x.grad.data.clamp_(-0.03, 0.03)
            adj.grad.data.clamp_(-0.03, 0.03)
            x = x - (step_size * x.grad) + noise_x
            adj = adj - (step_size * adj.grad) + noise_adj
            adj = (adj + torch.transpose(adj, 1, 2))/2

            x.data[:,:,0].clamp_(0, 1)
            x.data[:, :, 1:].clamp_(0, 28)
            adj.data.clamp_(0,1)

        x = x.detach()
        adj = adj.detach()


        # Reactivate gradients for parameters for training
        for p in model.parameters():
            p.requires_grad = True
        model.train(is_training)

        # Reset gradient calculation to setting before this function
        torch.set_grad_enabled(had_gradients_enabled)

        return DenseData(x, adj, mask)


