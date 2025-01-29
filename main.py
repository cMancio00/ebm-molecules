import torch
import matplotlib.pyplot as plt
import lightning as pl
import os
from models.ebm import DeepEnergyModel
from utils.Callback import GenerateCallback
import torchvision
from datetime import datetime

torch.set_float32_matmul_precision('high')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def cli_main():


        pl.seed_everything(42)

        model = DeepEnergyModel.load_from_checkpoint("lightning_logs/version_0/checkpoints/epoch=57-step=24882.ckpt")
        model.eval()
        callback = GenerateCallback(vis_steps=8, num_steps=1024, tensors_to_generate=4)

        imgs_per_step = callback.generate_imgs(model)
        imgs_per_step = imgs_per_step.cpu()
        
        grids = []
        for i in range(imgs_per_step.shape[1]):
                step_size = callback.num_steps // callback.vis_steps
                imgs_to_plot = imgs_per_step[step_size-1::step_size,i]
                imgs_to_plot = torch.cat([imgs_per_step[0:1,i],imgs_to_plot], dim=0)
                grid = torchvision.utils.make_grid(imgs_to_plot, nrow=imgs_to_plot.shape[0], normalize=True, pad_value=0.5, padding=2)
                grids.append(grid)
                
        final_grid = torchvision.utils.make_grid(grids, nrow=1, padding=2)
        final_grid = final_grid.permute(1, 2, 0)
        
        plt.figure(figsize=(8,8))
        plt.imshow(final_grid)
        plt.xlabel("Generation iteration")
        plt.xticks([(imgs_per_step.shape[-1]+2)*(0.5+j) for j in range(callback.vis_steps+1)],
                labels=[1] + list(range(step_size,imgs_per_step.shape[0]+1,step_size)))
        plt.yticks([])
        output_dir = "generated_images"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"./{output_dir}/generations_{timestamp}.png")


def main():
    pass


if __name__ == '__main__':
    cli_main()

