import torch
import matplotlib.pyplot as plt
import lightning as pl
import os
from models.ebm import DeepEnergyModel
from utils.Callback import GenerateCallback
import torchvision
import matplotlib.pyplot as plt
from DataModules import MNISTDataModule
from utils.Sampler import Sampler

torch.set_float32_matmul_precision('high')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():

        checkpoints = ["lightning_logs/version_0/checkpoints/epoch=57-step=24882.ckpt",
                       "lightning_logs/version_1/checkpoints/epoch=59-step=12840.ckpt"]
        seeds  = [1, 42, 43, 111]
        for seed in seeds:
            print(f"Using seed: {seed}")
            pl.seed_everything(seed)
            data = MNISTDataModule(batch_size=1024)
            data.prepare_data()
            data.setup("fit")
            for checkpoint in checkpoints:
                    print(f"\nUsing: {checkpoint}")
                    model_version = checkpoint.split("/")[1]
                    model = DeepEnergyModel.load_from_checkpoint(checkpoint)
                    model.eval()
                    training_energy = calculate_training_energy(data, model)
                    loop_generation(model, training_energy, model_version, seed)
                    steps_generation(model, model_version, seed)


def calculate_training_energy(data: pl.LightningDataModule, model: DeepEnergyModel):
        training_img, _ = next(iter(data.train_dataloader()))
        training_energy = -model.cnn(training_img.to(model.device)).mean()
        print(f"Training Energy: {training_energy:e}")
        return training_energy

def loop_generation(model: DeepEnergyModel, training_energy: float, model_version: str, seed: int, num_steps: int = 256, max_steps=10000):
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        for i in range(6):
                print(f"\nGenerating Image {i}")
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
                    if total_steps >= max_steps:
                            print(f"\nITERATION LIMIT REACHED, DISCARTING SAMPLE...")
                            break
        plt.suptitle(f"Training Energy: {training_energy:e}, Version: {model_version} Seed: {seed}", fontweight='bold')
        save_path = f"generated_images/loop/{model_version}"
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/seed_{seed}.png")

def steps_generation(model: DeepEnergyModel, model_version: str, seed: int):
    callback = GenerateCallback(vis_steps=8, num_steps=1024, tensors_to_generate=4)

    imgs_per_step = callback.generate_imgs(model)
    imgs_per_step = imgs_per_step.cpu()

    grids = []
    for i in range(imgs_per_step.shape[1]):
        step_size = callback.num_steps // callback.vis_steps
        imgs_to_plot = imgs_per_step[step_size - 1::step_size, i]
        imgs_to_plot = torch.cat([imgs_per_step[0:1, i], imgs_to_plot], dim=0)
        grid = torchvision.utils.make_grid(imgs_to_plot, nrow=imgs_to_plot.shape[0], normalize=True, pad_value=0.5,
                                           padding=2)
        grids.append(grid)

    final_grid = torchvision.utils.make_grid(grids, nrow=1, padding=2)
    final_grid = final_grid.permute(1, 2, 0)

    plt.figure(figsize=(8, 8))
    plt.imshow(final_grid)
    plt.xlabel(f"Generation iteration {model_version}, seed: {seed}")
    plt.xticks([(imgs_per_step.shape[-1] + 2) * (0.5 + j) for j in range(callback.vis_steps + 1)],
               labels=[1] + list(range(step_size, imgs_per_step.shape[0] + 1, step_size)))
    plt.yticks([])
    save_path = f"generated_images/steps/{model_version}"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/seed_{seed}.png")

if __name__ == '__main__':
    main()

