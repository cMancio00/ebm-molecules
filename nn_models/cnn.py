import torch
from torch import nn

class SmallCNN(nn.Module):

    IN_SHAPE = ()
    N_OUTPUTS = 0

    def __init__(self, conv_hidden_channels_list, full_hidden_neurons_list, **kwargs):
        super().__init__()

        conv_layers = nn.Sequential(
            nn.Conv2d(self.IN_SHAPE[0], conv_hidden_channels_list[0], kernel_size=5, stride=2, padding=4),
            nn.SiLU())
        for i in range(1, len(conv_hidden_channels_list)):
            conv_layers.append(
                nn.Conv2d(conv_hidden_channels_list[i - 1], conv_hidden_channels_list[i], kernel_size=3, stride=2, padding=1))
            conv_layers.append(nn.SiLU())
        conv_layers.append(nn.Flatten())

        fake_output = conv_layers(torch.randn(self.IN_SHAPE))
        dense_layers = nn.Sequential(nn.Linear(fake_output.numel(), full_hidden_neurons_list[0]),
                                     nn.SiLU())
        for i in range(1, len(full_hidden_neurons_list)):
            dense_layers.append(nn.Linear(full_hidden_neurons_list[i-1],full_hidden_neurons_list[i]))
            dense_layers.append(nn.SiLU())

        dense_layers.append(nn.Linear(full_hidden_neurons_list[-1], self.N_OUTPUTS))

        self.net = conv_layers + dense_layers

    def forward(self, x):
        x = self.net(x).squeeze(dim=-1)
        return x


class MNISTSmallCNN(SmallCNN):

    IN_SHAPE = (1, 28, 28)
    N_OUTPUTS = 10


class CIFAR10SmallCNN(SmallCNN):

    IN_SHAPE = (3, 32, 32)
    N_OUTPUTS = 10

