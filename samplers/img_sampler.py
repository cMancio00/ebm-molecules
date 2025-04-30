from typing import List, Any, Union
import torch
from .base import SamplerWithBuffer


class ImageSampler(SamplerWithBuffer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_shape = None

    def generate_samples(self, model, labels: torch.Tensor, start_point: Any = None,
                         steps: int = 60, step_size: float = 1.0) -> Any:

        if start_point is not None:
            x = start_point
        else:
            x = self.generate_random_samples(labels.shape[0], collate=True)
        x.detach_()
        x.requires_grad = True

        # save model state
        is_training = model.training
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        #############################################################

        noise_x = torch.randn(x.shape, device=labels.device)
        idx = torch.arange(labels.size(0), device=labels.device)

        # MCMC
        # batch.requires_grad = True
        for i in range(steps):
            noise_x.normal_(0, 0.005)
            x.data.add_(noise_x.data)
            x.data.clamp_(min=-1.0, max=1.0)
            energy = -model(x)[idx, labels]
            energy.sum().backward()
            x.grad.data.clamp_(-0.1, 0.1)
            x.data.add_(- (step_size * x.grad))
            x.grad.zero_()
            x.data.clamp_(min=-1.0, max=1.0)
        #############################################################

        # Reactivate gradients for parameters for training
        for p in model.parameters():
            p.requires_grad = True
        model.train(is_training)

        # Reset gradient calculation to setting before this function
        torch.set_grad_enabled(had_gradients_enabled)

        return x.detach()

    def generate_random_samples(self, num_samples: int, collate = True) -> Union[Any, List[Any]]:
        t = 2*torch.rand((num_samples,) + self.img_shape, device=self.device) - 1
        if collate:
            return t
        else:
            return list(torch.split(t, 1, dim=0))

    def collate_fn(self, data_list: List[Any]) -> Any:
        return torch.cat(data_list, dim=0)

