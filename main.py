import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import os
from models.ebm import DeepEnergyModel
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from utils.Callback import GenerateCallback, OutlierCallback, SamplerCallback
import torchvision
from DataModules import MNISTDataModule
from datetime import datetime

DATASET_PATH = "./datasets"
CHECKPOINT_PATH = "./saved_models/"

os.makedirs(CHECKPOINT_PATH, exist_ok=True)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():

    trainer = pl.Trainer(
                        max_epochs=5,
                        gradient_clip_val=0.1,
                        callbacks=[ModelCheckpoint(dirpath=CHECKPOINT_PATH,save_weights_only=True, mode="min", monitor='val_contrastive_divergence'),
                                GenerateCallback(every_n_epochs=1),
                                SamplerCallback(every_n_epochs=1),
                                OutlierCallback(),
                                LearningRateMonitor("epoch")
                                ])
    
    dm = MNISTDataModule(batch_size=256)
    model = DeepEnergyModel(img_shape=(1,28,28), lr=1e-4, beta1=0.0, batch_size=dm.batch_size)
    trainer.fit(model, dm)
    callback = GenerateCallback(batch_size=4, vis_steps=8, num_steps=256)
    imgs_per_step = callback.generate_imgs(model)
    imgs_per_step = imgs_per_step.cpu()
    
    for i in range(imgs_per_step.shape[1]):
        step_size = callback.num_steps // callback.vis_steps
        imgs_to_plot = imgs_per_step[step_size-1::step_size,i]
        imgs_to_plot = torch.cat([imgs_per_step[0:1,i],imgs_to_plot], dim=0)
        # grid = torchvision.utils.make_grid(imgs_to_plot, nrow=imgs_to_plot.shape[0], normalize=True, range=(-1,1), pad_value=0.5, padding=2)
        grid = torchvision.utils.make_grid(imgs_to_plot, nrow=imgs_to_plot.shape[0], normalize=True, pad_value=0.5, padding=2)
        
        grid = grid.permute(1, 2, 0)
        plt.figure(figsize=(8,8))
        plt.imshow(grid)
        plt.xlabel("Generation iteration")
        plt.xticks([(imgs_per_step.shape[-1]+2)*(0.5+j) for j in range(callback.vis_steps+1)],
                labels=[1] + list(range(step_size,imgs_per_step.shape[0]+1,step_size)))
        plt.yticks([])
        output_dir = "generated_images"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"./{output_dir}/generations_{timestamp}_{i}.png")


if __name__ == '__main__':
    main()


