import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import argparse
import pytorch_lightning as pl
import os
from models.ebm import DeepEnergyModel
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from utils.Callback import GenerateCallback, OutlierCallback, SamplerCallback
import torchvision

print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

DATASET_PATH = "./datasets"
CHECKPOINT_PATH = "./saved_models/"

os.makedirs(CHECKPOINT_PATH, exist_ok=True)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def prepare_data(train_batch_size, test_batch_size, validation_split):
    
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])
    dataset = datasets.MNIST('./datasets', train=True, download=True, transform=transform)
    validation_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - validation_size
    train_set, validation_set = random_split(dataset, [train_size, validation_size])
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True,  drop_last=True,  num_workers=4, pin_memory=True)
    validation_loader = DataLoader(validation_set, batch_size=128, shuffle=False)
    
    test_set = datasets.MNIST('./datasets', train=False, transform=transform)
    test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False, drop_last=False, num_workers=4)
    return train_loader, validation_loader, test_loader

def train_model(train_loader, test_loader, **kwargs):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "MNIST"),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=60,
                         gradient_clip_val=0.1,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor='val_contrastive_divergence'),
                                    GenerateCallback(every_n_epochs=5),
                                    SamplerCallback(every_n_epochs=5),
                                    OutlierCallback(),
                                    LearningRateMonitor("epoch")
                                   ])
    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "MNIST.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = DeepEnergyModel.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)
        model = DeepEnergyModel(**kwargs)
        trainer.fit(model, train_loader, test_loader)
        model = DeepEnergyModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    # No testing as we are more interested in other properties
    return model

def main():
    parser = argparse.ArgumentParser(description='MNIST')
    parser.add_argument('--train-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--validation-split', type=float, default=0.15, metavar='V',
                        help='input size to use in validation (default: 0.15)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)


    train_loader, validation_loader, test_loader = prepare_data(args.train_batch_size, args.test_batch_size, args.validation_split)

    print(f'Train dataset size: {len(train_loader.dataset)}')
    print(f'Validation dataset size: {len(validation_loader.dataset)}')
    print(f'Test dataset size: {len(test_loader.dataset)}')
    
    # model = train_model(train_loader, test_loader, img_shape=(1,28,28),
    #                 batch_size=train_loader.batch_size,
    #                 lr=1e-4,
    #                 beta1=0.0)
    
    model = DeepEnergyModel.load_from_checkpoint("epoch=36-step=14726.ckpt")
    model.to(device)
    pl.seed_everything(43)
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
    plt.savefig("./generations.png")






if __name__ == '__main__':
    main()


