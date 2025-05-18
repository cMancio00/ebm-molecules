import torch
from torch import nn, sigmoid

class Swish(nn.Module):

    def forward(self, x):
        return x * sigmoid(x)
    

class SmallCNN(nn.Module):

    IN_SHAPE = ()
    N_OUTPUTS = 0

    def __init__(self, hidden_features, **kwargs):
        super().__init__()
        c_hid1 = hidden_features//2
        c_hid2 = hidden_features
        c_hid3 = hidden_features*2

        conv_layers = nn.Sequential(
            nn.Conv2d(self.IN_SHAPE[0], c_hid1, kernel_size=5, stride=2, padding=4),
            Swish(),
            nn.Conv2d(c_hid1, c_hid2, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(c_hid2, c_hid3, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(c_hid3, c_hid3, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Flatten()
        )
        fake_output = conv_layers(torch.randn(self.IN_SHAPE))
        dense_layers = nn.Sequential(
                nn.Linear(fake_output.numel(), c_hid3),
                Swish(),
                nn.Linear(c_hid3, self.N_OUTPUTS)
        )
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

