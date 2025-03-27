import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from models.graph_models import MoNet, gcn
from utils.Sampler import Sampler
import lightning as pl
import torch.optim as optim
from utils.graphs import generate_random_graph
from torch_geometric.data import Batch

class DeepEnergyModel(pl.LightningModule):

    def __init__(self, batch_size : int = 32, alpha=0.1, lr=1e-4, beta1=0.0, mcmc_steps: int = 60, mcmc_learning_rate: float = 10.0):
        super().__init__()
        self.save_hyperparameters()
        # self.cnn = MoNet()
        self.cnn = gcn()
        # self.cnn: pl.LightningModule = MoNet().to(self.device)
        self.batch_size = batch_size
        self.sampler = Sampler(self.cnn, sample_size=self.batch_size)
        self.mcmc_steps = mcmc_steps
        self.mcmc_learning_rate = mcmc_learning_rate

    def forward(self, x):
        z = self.cnn(x)
        return z

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.97)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        labels: Tensor = batch.y
        positive_energy: Tensor = self.cnn(batch)
        generated_samples: Batch = self.sampler.sample_new_tensor(steps=self.mcmc_steps, step_size=self.mcmc_learning_rate, labels=labels)
        negative_energy: Tensor = self.cnn(generated_samples)

        cross_entropy: Tensor = CrossEntropyLoss()(positive_energy, labels)

        positive_energy = positive_energy[torch.arange(labels.size(0)), labels]
        negative_energy = negative_energy[torch.arange(labels.size(0)),labels]
        generative_loss: Tensor = (negative_energy - positive_energy).mean()
        loss: Tensor = generative_loss

        self.log('loss', loss, batch_size=self.batch_size)
        # self.log('loss_contrastive_divergence', generative_loss)
        # self.log('Positive_phase_energy', positive_energy.mean())
        # self.log('Negative_phase_energy', negative_energy.mean())
        # self.log("Cross Entropy", cross_entropy)
        return loss
    
    def validation_step(self, batch, batch_idx):
        labels: Tensor = batch.y
        positive_energy: Tensor = self(batch)
        random_noise: Batch = Batch.from_data_list([generate_random_graph(device=self.device) for _ in range(self.batch_size)])
        negative_energy: Tensor = self(random_noise)

        positive_energy = positive_energy[torch.arange(labels.size(0)), labels]
        negative_energy = negative_energy[torch.arange(labels.size(0)),labels]

        loss: Tensor = (negative_energy - positive_energy).mean()
        self.log('val_contrastive_divergence', loss, batch_size=self.batch_size)
        # We can add CrossEntropy in validation
        return loss


    def on_load_checkpoint(self, checkpoint):
        state_dict = checkpoint["state_dict"]
        new_state_dict = {}
        for key, value in state_dict.items():
            if "parametrizations.weight.original" in key:
                new_key = key.replace("parametrizations.weight.original", "weight")
                new_state_dict[new_key] = value
            elif "parametrizations.weight.0._u" in key or "parametrizations.weight.0._v" in key:
                continue
            else:
                new_state_dict[key] = value
        checkpoint["state_dict"] = new_state_dict

    def on_train_start(self) -> None:
        self.sampler.init_buffer()
        