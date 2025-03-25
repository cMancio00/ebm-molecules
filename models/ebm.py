import torch
from torch.nn import CrossEntropyLoss
from models import Small_CNN
from models.graph_models import MoNet
from utils.Sampler import Sampler
import lightning as pl
import torch.optim as optim
from utils.graphs import generate_random_graph, concat_batches
from torch_geometric.data import Data, Batch

class DeepEnergyModel(pl.LightningModule):

    def __init__(self, batch_size : int = 32, alpha=0.1, lr=1e-4, beta1=0.0, mcmc_steps: int = 60, mcmc_learning_rate: float = 10.0,**CNN_args):
        super().__init__()
        self.save_hyperparameters()
        self.cnn: pl.LightningModule = MoNet(**CNN_args)
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

    # def training_step(self, batch, batch_idx):
    #     # A Data in the Batch will have the following structure:
    #     # Data(x=[75, 1], edge_index=[2, 1431], y=[1], pos=[75, 2], edge_attr=[1431, 2])
    #     labels: torch.Tensor = batch.y
    #     batch.__delattr__('y')
    #     cross_entropy = CrossEntropyLoss()(self.cnn(batch), labels)
    #
    #     # -----Sample------
    #     fake_imgs = self.sampler.sample_new_tensor(steps=self.mcmc_steps, step_size=self.mcmc_learning_rate, labels=labels)
    #
    #     print(f"\nBatch type: {type(batch)}")
    #     print(batch[0])
    #     print(batch)
    #     print(f"Samples type: {type(fake_imgs)}")
    #     print(fake_imgs)
    #
    #     # Predict energy score for all images
    #     # inp_imgs = torch.cat([batch, fake_imgs], dim=0)
    #     # inp_imgs = concat_batches([batch, fake_imgs])
    #     # real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim=0)
    #     # real_out = real_out[torch.arange(labels.size(0)),labels]
    #     # fake_out = fake_out[torch.arange(labels.size(0)),labels]
    #
    #     real_out = self.cnn(batch)
    #     fake_out = self.cnn(fake_imgs)
    #     real_out = real_out[torch.arange(labels.size(0)),labels]
    #     fake_out = fake_out[torch.arange(labels.size(0)),labels]
    #
    #     # Calculate losses
    #     reg_loss = 0 # self.hparams.alpha * (real_out ** 2 + fake_out ** 2).mean()
    #     generative_loss = (fake_out - real_out).mean()
    #     # loss = cross_entropy + generative_loss + reg_loss
    #     loss = generative_loss
    #     # Logging
    #     self.log('loss', loss)
    #     self.log('loss_regularization', reg_loss)
    #     self.log('loss_contrastive_divergence', generative_loss)
    #     self.log('Positive_phase_energy', real_out.mean())
    #     self.log('Negative_phase_energy', fake_out.mean())
    #     self.log("Cross Entropy", cross_entropy)
    #     return loss

    def training_step(self, batch, batch_idx):
        labels = batch.y
        positive_energy = self(batch)
        generated_samples = self.sampler.sample_new_tensor(steps=self.mcmc_steps, step_size=self.mcmc_learning_rate, labels=labels)
        negative_energy = self(generated_samples)

        positive_energy = positive_energy[torch.arange(labels.size(0)), labels]
        negative_energy = negative_energy[torch.arange(labels.size(0)),labels]
        return (negative_energy - positive_energy).mean()
    
    def validation_step(self, batch, batch_idx):
        # # Da rivedere
        # labels: torch.Tensor = batch.y
        # # Bisogna inserire il batch_size
        # fake_imgs = [generate_random_graph() for _ in range(self.batch_size)]
        #
        # inp_imgs = torch.cat([batch, fake_imgs], dim=0)
        # real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim=0)
        # real_out = real_out[torch.arange(labels.size(0)),labels]
        # fake_out = fake_out[torch.arange(labels.size(0)),labels]
        #
        # # cdiv = fake_out.mean() - real_out.mean()
        # cdiv = (fake_out - real_out).mean()
        # self.log('val_contrastive_divergence', cdiv)
        pass

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


        