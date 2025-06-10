from typing import List, Any, Union, Tuple
import torch
from matplotlib import pyplot as plt
from torch import nn
from .base import SamplerWithBuffer


class ImageSampler(SamplerWithBuffer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_shape = None

    def _MCMC_generation(self, model: nn.Module, steps: int, step_size: float, labels: torch.Tensor, starting_x: Any,
                         is_training) -> Any:

        x = starting_x
        x.detach_()
        x.requires_grad = True

        noise_x = torch.randn(x.shape, device=labels.device)
        idx = torch.arange(labels.size(0), device=labels.device)

        # MCMC
        for i in range(steps):
            noise_x.normal_(0, 0.01)
            energy = -model(x)[idx, labels]
            energy.sum().backward()
            x.data.add_(- (step_size * x.grad) + noise_x)
            x.grad.zero_()

        return x.detach()

    @torch.no_grad()
    def generate_random_batch(self, batch_size, device=None, collate=True):
        if device is None:
            device = self.device

        rand_x = torch.randn((batch_size,) + self.img_shape, device=device)
        rand_y = torch.randint(0, self.num_classes, (batch_size,), device=device)

        if collate:
            return rand_x, rand_y
        else:
            return [(rand_x[i], rand_y[i]) for i in range(batch_size)]

    @staticmethod
    def collate_fn(data_list: List[Tuple[Any, torch.Tensor]]) -> Tuple[Any, torch.Tensor]:
        x_list, y_list = zip(*data_list)
        return torch.stack(x_list), torch.stack(y_list)

    def plot_sample(self, s: Tuple[torch.Tensor, torch.Tensor], ax: plt.Axes) -> None:
        img = s[0]
        low, high = img.min(), img.max()
        img.sub_(low)
        img.div_(max(high - low, 1e-5))
        ax.imshow(img.permute(1, 2, 0), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Label {s[1]}')
