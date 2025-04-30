from data_modules import MNISTSuperpixelDataModule
from ebm.ebm import DeepEnergyModel
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from utils.callbacks import BufferSamplerCallback
from torch import set_float32_matmul_precision
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import seed_everything

set_float32_matmul_precision('high')

seed_everything(0)
batch_size: int = 128
data_module = MNISTSuperpixelDataModule(
    num_workers=0,
    batch_size=batch_size,
    filter_target=[0,1]
)
trainer = Trainer(
    default_root_dir="graph_logs",
    logger=TensorBoardLogger("graph_logs"),
    max_epochs=101,
    callbacks=[
        ModelCheckpoint(),
        BufferSamplerCallback(num_samples=32,num_rows=8,every_n_epochs=5)
    ])

model = DeepEnergyModel(batch_size=batch_size, mcmc_steps=20, mcmc_learning_rate=10)
# model = GCN_Dense(3,32,10)
trainer.fit(model, data_module)
