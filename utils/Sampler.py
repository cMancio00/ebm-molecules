import random
import torch
import numpy as np
from lightning import LightningModule

class Sampler:

    def __init__(self, model: LightningModule, img_shape: tuple[int, int, int], sample_size: int, max_len: int=8192):
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
        self.buffer = [(torch.rand((1,)+img_shape)*2-1) for _ in range(self.sample_size)]
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def sample_new_exmps(self, steps: int=60, step_size: int=10):
        """
        Function for getting a new batch of "fake" images.
        Inputs:
            steps - Number of iterations in the MCMC algorithm
            step_size - Learning rate nu in the algorithm above
        """
        # Choose 95% of the batch from the buffer, 5% generate from scratch
        num_new_samples = np.random.binomial(self.sample_size, 0.05)
        new_rand_tensors = torch.rand((num_new_samples,) + self.img_shape) * 2 - 1
        # Randomlly select the old tensor from the buffer
        old_tensors = torch.cat(random.choices(self.buffer, k=self.sample_size-num_new_samples), dim=0)
        # Concatenate the old tensors and the new ones in the batch dimension
        mcmc_starting_tensors = torch.cat([new_rand_tensors, old_tensors], dim=0).detach().to(self.device)

        # Perform MCMC sampling
        mcmc_samples = Sampler.generate_samples(self.model, mcmc_starting_tensors, steps=steps, step_size=step_size)

        # Add new images to the buffer and remove old ones if needed
        self.buffer = list(mcmc_samples.to(torch.device("cpu")).chunk(self.sample_size, dim=0)) + self.buffer
        self.buffer = self.buffer[:self.max_len]
        return mcmc_samples

    @staticmethod
    def generate_samples(model, inp_imgs, steps=60, step_size=10, return_img_per_step=False):
        """
        Function for sampling images for a given model.
        Inputs:
            model - Neural network to use for modeling E_theta
            inp_imgs - Images to start from for sampling. If you want to generate new images, enter noise between -1 and 1.
            steps - Number of iterations in the MCMC algorithm.
            step_size - Learning rate nu in the algorithm above
            return_img_per_step - If True, we return the sample at every iteration of the MCMC
        """
        # Before MCMC: set model parameters to "required_grad=False"
        # because we are only interested in the gradients of the input.
        is_training = model.training
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        inp_imgs.requires_grad = True

        # Enable gradient calculation if not already the case
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # We use a buffer tensor in which we generate noise each loop iteration.
        # More efficient than creating a new tensor every iteration.
        noise = torch.randn(inp_imgs.shape, device=inp_imgs.device)

        # List for storing generations at each step (for later analysis)
        imgs_per_step = []

        # Loop over K (steps)
        for _ in range(steps):
            # Part 1: Add noise to the input.
            noise.normal_(0, 0.005)
            inp_imgs.data.add_(noise.data)
            inp_imgs.data.clamp_(min=-1.0, max=1.0)

            # Part 2: calculate gradients for the current input.
            out_imgs = -model(inp_imgs)
            out_imgs.sum().backward()
            inp_imgs.grad.data.clamp_(-0.03, 0.03)

            # Apply gradients to our current samples
            inp_imgs.data.add_(-step_size * inp_imgs.grad.data)
            inp_imgs.grad.detach_()
            inp_imgs.grad.zero_()
            inp_imgs.data.clamp_(min=-1.0, max=1.0)

            if return_img_per_step:
                imgs_per_step.append(inp_imgs.clone().detach())

        # Reactivate gradients for parameters for training
        for p in model.parameters():
            p.requires_grad = True
        model.train(is_training)

        # Reset gradient calculation to setting before this function
        torch.set_grad_enabled(had_gradients_enabled)

        if return_img_per_step:
            return torch.stack(imgs_per_step, dim=0)
        else:
            return inp_imgs