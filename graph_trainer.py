import torch
from torch_geometric.data import Batch
from DataModules import MNISTSuperpixelDataModule
from models.ebm import DeepEnergyModel
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from torch import set_float32_matmul_precision
from lightning.pytorch.loggers import TensorBoardLogger
from utils import Sampler
from utils.graphs import superpixels_to_2d_image, generate_random_graph
import matplotlib.pyplot as plt


set_float32_matmul_precision('high')

class UploadTrainingImagesCallback(Callback):
    def __init__(self, every_n_epochs: int=1, images_to_upload: int = 4, dpi: int = 200):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.images_to_upload = images_to_upload
        self.dpi = dpi

    def plot_graph(self, batch):
        fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=self.dpi)
        axes = axes.flatten()
        print(batch)
        for i in range(len(axes)):
            image = superpixels_to_2d_image(batch[i])

            print(image.shape)
            axes[i].imshow(image, cmap=plt.cm.binary)
            axes[i].axis("off")
            # axes[i].set_title(f"{batch[i].y.item()}")
        return fig

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:

            labels = torch.randint(0, 2, (self.images_to_upload,))
            print(f"Generating...")

            starting = [generate_random_graph(device=pl_module.device) for _ in range(self.images_to_upload)]
            starting = Batch.from_data_list(starting)
            generated_samples = Sampler.generate_samples(pl_module.cnn, starting, torch.randint(0, 2, (1,)),
                                                 256, 1)
            print(generated_samples)

            print(f"Updating {self.images_to_upload} images to Tensorboard")
            grid = self.plot_graph(generated_samples)
            trainer.logger.experiment.add_figure(f"Data Batch Graphs", grid, global_step=trainer.current_epoch)
            print(f"Done updating {self.images_to_upload} images to Tensorboard")


batch_size: int = 128
data_module = MNISTSuperpixelDataModule(num_workers=4, batch_size=batch_size)
trainer = Trainer(
    default_root_dir="graph_logs",
    logger=TensorBoardLogger("graph_logs"),
    max_epochs=51,
    callbacks=[
        ModelCheckpoint(),
        # UploadTrainingImagesCallback()
    ])

model = DeepEnergyModel(batch_size=batch_size, mcmc_steps=20)
# model = GCN_Dense(
#     in_channels=1,
#     hidden_channels=64,
#     out_channels=10,
# )
trainer.fit(model, data_module)
