import torch
import matplotlib.pyplot as plt
import lightning as pl
import os
from models.ebm import DeepEnergyModel
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from utils.Callback import GenerateCallback, OutlierCallback, SamplerCallback
import torchvision
from DataModules import MNISTDataModule
from datetime import datetime
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import TensorBoardLogger

CHECKPOINT_PATH = "./saved_models/"
torch.set_float32_matmul_precision('high')
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def cli_main():
        pl.seed_everything(42)
        logger = TensorBoardLogger("tb_logs")
        trainer = pl.Trainer(
                max_epochs=60,
                gradient_clip_val=0.1,
                logger=logger,
                callbacks=[ModelCheckpoint(dirpath=CHECKPOINT_PATH,save_weights_only=True, mode="min", monitor='val_contrastive_divergence'),
                        GenerateCallback(every_n_epochs=1),
                        SamplerCallback(every_n_epochs=1),
                        OutlierCallback(),
                        LearningRateMonitor("epoch")
                        ])
        dm = MNISTDataModule(batch_size=128)
        model = DeepEnergyModel(img_shape=(1,28,28), lr=1e-4, beta1=0.0, batch_size=dm.batch_size)
        trainer.fit(model=model,datamodule=dm)
        
        # cli = LightningCLI(DeepEnergyModel, MNISTDataModule)

        pl.seed_everything(42)
        
        # model = DeepEnergyModel.load_from_checkpoint("epoch=36-step=14726.ckpt")
        callback = GenerateCallback(vis_steps=8, num_steps=256)
        # imgs_per_step = callback.generate_imgs(cli.model)
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

