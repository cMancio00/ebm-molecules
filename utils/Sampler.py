import random
import torch
import numpy as np
from lightning import LightningModule


class Sampler:

    def __init__(self, model: LightningModule, img_shape: tuple[int, int, int], sample_size: int, max_len: int = 8192):
        """
        Inputs:
            model - Neural network to use for modeling E_theta
            img_shape - Shape of the images to model
            sample_size - Batch size of the samples
            max_len - Maximum number of data points to keep in the buffer
        """
        super().__init__()
        self.model = model
        self.img_shape = tuple(img_shape)
        self.sample_size = sample_size
        self.max_len = max_len
        self.buffer = [(torch.rand((1,) + img_shape) * 2 - 1) for _ in range(self.sample_size)]
        # self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def sample_new_tensor(self, steps: int = 60, step_size: int = 10):
        """
        Function for getting a new batch of sampled tensors via MCMC.
        Inputs:
            steps - Number of iterations in the MCMC.
            step_size - Learning rate nu in the algorithm above
        """
        # Choose 95% of the batch from the buffer, 5% generate from scratch
        num_new_samples = np.random.binomial(self.sample_size, 0.05)
        new_rand_tensors = torch.rand((num_new_samples,) + self.img_shape) * 2 - 1
        # Randomlly select the old tensor from the buffer
        old_tensors = torch.cat(random.choices(self.buffer, k=self.sample_size - num_new_samples), dim=0)
        # Concatenate the old tensors and the new ones in the batch dimension
        mcmc_starting_tensors = torch.cat([new_rand_tensors, old_tensors], dim=0).detach().to(self.model.device)

        # Perform MCMC sampling
        mcmc_samples = Sampler.generate_samples(self.model, mcmc_starting_tensors, steps=steps, step_size=step_size)

        # Add new images to the buffer and remove old ones if needed
        self.buffer = list(mcmc_samples.to(torch.device("cpu")).chunk(self.sample_size, dim=0)) + self.buffer
        self.buffer = self.buffer[:self.max_len]
        return mcmc_samples

    @staticmethod
    def generate_samples(model: LightningModule, mcmc_starting_tensors: torch.Tensor, steps: int = 60, step_size: float = 10.0,
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
            # inplace methods, can't compile the model with them
            mcmc_starting_tensors.data.add_(noise.data)
            mcmc_starting_tensors.data.clamp_(min=-1.0, max=1.0)

            # Part 2: Get the energy and calculate the gradient of the input
            energy: torch.Tensor = -model(mcmc_starting_tensors)
            energy.sum().backward()
            # inplace method
            # Do we need this?
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
