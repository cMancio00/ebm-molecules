from DataModules import MNISTSuperpixelDataModule
from models.ebm import DeepEnergyModel
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import set_float32_matmul_precision
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import seed_everything

set_float32_matmul_precision('high')

seed_everything(0)
batch_size: int = 128
data_module = MNISTSuperpixelDataModule(
    num_workers=6,
    batch_size=batch_size,
    filter_target=[0,1]
)
trainer = Trainer(
    default_root_dir="graph_logs",
    logger=TensorBoardLogger("graph_logs"),
    max_epochs=101,
    callbacks=[
        ModelCheckpoint(),
    ])

model = DeepEnergyModel(batch_size=batch_size, mcmc_steps=20)
trainer.fit(model, data_module)
