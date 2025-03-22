from DataModules import MNISTSuperpixelDataModule
from models.graph_models import MoNet
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import set_float32_matmul_precision
from lightning.pytorch.loggers import TensorBoardLogger

set_float32_matmul_precision('high')

data_module = MNISTSuperpixelDataModule()
trainer = Trainer(
    default_root_dir="graph_logs",
    logger=TensorBoardLogger("lightning_logs"),
    max_epochs=10,
    callbacks=[
    ModelCheckpoint()
    ])
model = MoNet()
trainer.fit(model, data_module)
