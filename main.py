import torch
import matplotlib.pyplot as plt
import lightning as pl
import os
from models.ebm import DeepEnergyModel
from utils.Callback import GenerateCallback
import torchvision
from datetime import datetime
import matplotlib.pyplot as plt
from DataModules import MNISTDataModule
from utils.Sampler import Sampler

torch.set_float32_matmul_precision('high')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def cli_main():


        pl.seed_everything(111)

        model = DeepEnergyModel.load_from_checkpoint("lightning_logs/version_0/checkpoints/epoch=57-step=24882.ckpt")
        model.eval()
        # callback = GenerateCallback(vis_steps=8, num_steps=512, tensors_to_generate=6)


        data = MNISTDataModule(batch_size=1024)
        data.prepare_data()
        data.setup("fit")

        training_img, _ = next(iter(data.train_dataloader()))
        training_energy = -model.cnn(training_img.to(model.device)).mean()
        print(f"Training Energy: {training_energy:e}")

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        for i in range(6):
                print(f"\nGenerating Image {i}")
                num_steps = 256
                total_steps = num_steps
                generated = (torch.rand((1,1,28,28)) * 2 - 1).to(model.device)
                while True:
                    generated = Sampler.generate_samples(model.cnn, generated, num_steps, model.mcmc_learning_rate)
                    energy = -model.cnn(generated).item()
                    if energy < training_energy:
                        print(f"\nSample energy: {energy:e} less then Training Energy: {training_energy:e}, VALID")
                        axes[i].imshow(generated[0].cpu().detach().permute(1,2,0), cmap='gray')
                        axes[i].axis('off')
                        axes[i].set_title(f"Energy: {energy:e}", fontweight='bold')
                        break
                    total_steps += num_steps
                    print(f"\rEnergy: {energy:e} is too high, adding {num_steps} iterations (total {total_steps})",end='', flush=True)
                    if total_steps >= 10000:
                            print(f"\nITERATION LIMIT REACHED, DISCARTING SAMPLE...")
                            break
        plt.suptitle(f"Training Energy: {training_energy:e}", fontweight='bold')
        plt.savefig("Generated loop Images.png")

        # num_images = callback.tensors_to_generate
        # fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        # axes = axes.flatten()
        # for i in range(num_images):
        #         imgs_per_step = callback.generate_imgs(model)[callback.num_steps - 1, i]
        #         energy = model.cnn(imgs_per_step.unsqueeze(0)).item()
        #         axes[i].imshow(imgs_per_step.cpu().permute(1, 2, 0), cmap='gray')
        #         axes[i].axis('off')
        #         axes[i].set_title(f"Energy: {energy:e}", fontweight='bold')
        # plt.suptitle(f"Training Energy: {training_energy:e}", fontweight='bold')
        # plt.savefig("Generated Images.png")



        # imgs_per_step = imgs_per_step.cpu()
        #
        # grids = []
        # for i in range(imgs_per_step.shape[1]):
        #         step_size = callback.num_steps // callback.vis_steps
        #         imgs_to_plot = imgs_per_step[step_size-1::step_size,i]
        #         imgs_to_plot = torch.cat([imgs_per_step[0:1,i],imgs_to_plot], dim=0)
        #         grid = torchvision.utils.make_grid(imgs_to_plot, nrow=imgs_to_plot.shape[0], normalize=True, pad_value=0.5, padding=2)
        #         grids.append(grid)
        #
        # final_grid = torchvision.utils.make_grid(grids, nrow=1, padding=2)
        # final_grid = final_grid.permute(1, 2, 0)
        #
        # plt.figure(figsize=(8,8))
        # plt.imshow(final_grid)
        # plt.xlabel("Generation iteration")
        # plt.xticks([(imgs_per_step.shape[-1]+2)*(0.5+j) for j in range(callback.vis_steps+1)],
        #         labels=[1] + list(range(step_size,imgs_per_step.shape[0]+1,step_size)))
        # plt.yticks([])
        # output_dir = "generated_images"
        # os.makedirs(output_dir, exist_ok=True)
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # plt.savefig(f"./{output_dir}/generations_{timestamp}.png")


def main():
    pass


if __name__ == '__main__':
    cli_main()

