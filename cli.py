import torch
from models.ebm import DeepEnergyModel
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from utils.Callback import GenerateCallback, SamplerCallback, SpectralNormalizationCallback
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import TensorBoardLogger

torch.set_float32_matmul_precision('high')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("model.batch_size", "data.init_args.batch_size")

def cli_main():

    cli = MyLightningCLI(
        DeepEnergyModel,
        seed_everything_default=42,
        trainer_defaults={
            'max_epochs': 61,
            'gradient_clip_val': 0.1,
            'callbacks': [
                ModelCheckpoint(save_top_k=1,auto_insert_metric_name=True,
                                monitor='val_contrastive_divergence'),
                GenerateCallback(every_n_epochs=5, num_steps=1024, vis_steps=10),
                SamplerCallback(every_n_epochs=5),
                # SpectralNormalizationCallback(),
                LearningRateMonitor("epoch")
            ]
        }
    )

    cli.trainer.logger = TensorBoardLogger("lightning_logs")

if __name__ == '__main__':
    cli_main()

