from torch import nn, hub


class HubModel(nn.Module):

    def __init__(self, repo_name, model_name, pretrained, **kwargs):
        super().__init__()
        self.model = hub.load(repo_or_dir=repo_name, model=model_name, pretrained=pretrained, **kwargs)

    def forward(self, x):
        return self.model(x)
