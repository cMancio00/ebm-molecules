from torch import nn

class SmallCNN(nn.Module):
    
    def __init__(self, hidden_features, out_dim, **kwargs):
        super().__init__()
        c_hid1 = hidden_features//2
        c_hid2 = hidden_features
        c_hid3 = hidden_features*2

        self.cnn_layers = nn.Sequential(
                nn.Conv2d(1, c_hid1, kernel_size=5, stride=2, padding=4),
                nn.SiLU(),
                nn.Conv2d(c_hid1, c_hid2, kernel_size=3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(c_hid2, c_hid3, kernel_size=3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(c_hid3, c_hid3, kernel_size=3, stride=2, padding=1),
                nn.SiLU(),
                nn.Flatten(),
                nn.Linear(c_hid3*4, c_hid3),
                nn.SiLU(),
                nn.Linear(c_hid3, out_dim)
        )

    def forward(self, x):
        x = self.cnn_layers(x).squeeze(dim=-1)
        return x
