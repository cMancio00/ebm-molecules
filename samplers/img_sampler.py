from typing import List, Any, Union, Tuple
import torch
from torch import nn
from .base import SamplerWithBuffer


class ImageSampler(SamplerWithBuffer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_shape = None

    def _MCMC_generation(self, model: nn.Module, steps: int, step_size: float, labels: torch.Tensor, starting_x: Any) -> Any:

        x = starting_x
        x.detach_()
        x.requires_grad = True

        noise_x = torch.randn(x.shape, device=labels.device)
        idx = torch.arange(labels.size(0), device=labels.device)

        # MCMC
        for i in range(steps):
            noise_x.normal_(0, 0.005)
            x.data.clamp_(min=-1.0, max=1.0)
            energy = -model(x)[idx, labels]
            energy.sum().backward()
            x.grad.data.clamp_(-0.1, 0.1)
            x.data.add_(- (step_size * x.grad) + noise_x)
            x.grad.zero_()
            x.data.clamp_(min=-1.0, max=1.0)

        return x.detach()

    def generate_random_batch(self, batch_size, device=None, collate=True):
        if device is None:
            device = self.device

        rand_x = 2*torch.rand((batch_size,) + self.img_shape, device=device) - 1
        rand_y = torch.randint(0, self.num_classes, (batch_size,), device=device)

        if collate:
            return rand_x, rand_y
        else:
            return [(rand_x[i], rand_y[i]) for i in range(batch_size)]

    @staticmethod
    def collate_fn(data_list: List[Tuple[Any, torch.Tensor]]) -> Tuple[Any, torch.Tensor]:
        x_list, y_list = zip(*data_list)
        return torch.stack(x_list), torch.stack(y_list)
