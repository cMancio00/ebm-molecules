from DataModules import MNISTSuperpixelDataModule
from models.ebm import DeepEnergyModel
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from torch import set_float32_matmul_precision, device, cuda
from lightning.pytorch.loggers import TensorBoardLogger
from models.graph_models import MoNet
from utils import Sampler
from utils.graphs import superpixels_to_2d_image
import matplotlib.pyplot as plt


set_float32_matmul_precision('high')

class UploadTrainingImagesCallback(Callback):
    def __init__(self, every_n_epochs: int=5, images_to_upload: int = 4, dpi: int = 200):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.images_to_upload = images_to_upload
        self.dpi = dpi

    def plot_graph(self, batch):
        fig, axes = plt.subplots(1, self.images_to_upload, figsize=(10, 6), dpi=self.dpi)
        axes = axes.flatten()
        for i in range(len(axes)):
            image = superpixels_to_2d_image(batch[i])
            print(image.shape)
            axes[i].imshow(image, cmap=plt.cm.binary)
            axes[i].axis("off")
            axes[i].set_title(f"{batch[i].y.item()}")
        return fig

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            if trainer.logger is not None:
                print(f"Updating {self.images_to_upload} images to Tensorboard")
                grid = self.plot_graph(trainer.train_dataloader.dataset[:self.images_to_upload])
                trainer.logger.experiment.add_figure(f"Data Batch Graphs", grid, global_step=trainer.current_epoch)
                print(f"Done updating {self.images_to_upload} images to Tensorboard")

data_module = MNISTSuperpixelDataModule(num_workers=4, batch_size=64)
trainer = Trainer(
    default_root_dir="graph_logs",
    logger=TensorBoardLogger("graph_logs"),
    max_epochs=11,
    callbacks=[
        ModelCheckpoint(),
        # UploadTrainingImagesCallback()
    ])

model = DeepEnergyModel(batch_size=64, mcmc_steps=20)
trainer.fit(model, data_module)
