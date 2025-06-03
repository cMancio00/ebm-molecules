import torch
from torch import nn
from torch_geometric.nn import Linear
from torch_geometric.nn.dense import DenseGCNConv
from torch_geometric.typing import OptTensor


class MultiEdgeDenseGCNConv(DenseGCNConv):
    r"""
        DenseGCNConv with accept multi-dimensional adjacency matrix, i.e. A has shape N x N x F, but all the values are
        between 0 and 1
    """

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: OptTensor = None,
                add_loop: bool = True) -> torch.Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
            adj (torch.Tensor): Adjacency tensor
                :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N \times F}`,
                The adjacency tensor is broadcastable in the batch dimension,
                resulting in a shared adjacency matrix for the complete batch.
            mask (torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
            add_loop (bool, optional): If set to :obj:`False`, the layer will
                not automatically add self-loops to the adjacency matrices.
                (default: :obj:`True`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 3 else adj
        B, N, _, F = adj.size()  # has shape B x N x N x F

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx, :] = 1 if not self.improved else 2

        out = self.lin(x)
        deg_inv_sqrt = adj.sum(dim=2).clamp(min=1).pow(-0.5) # has shape B x N x F

        adj = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(2) # has shape B x N x N x F
        out = (adj.unsqueeze(-1) * out.view(B, 1, N, 1, -1)).sum(dim=(2, 3))  # has shape B x F x N x out_channels

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out


class DenseGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels_list, out_channels, use_edge_types=False):
        super().__init__()

        conv_class = MultiEdgeDenseGCNConv if use_edge_types else DenseGCNConv

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(conv_class(in_channels, hidden_channels_list[0]))
        for i in range(1, len(hidden_channels_list)):
            self.conv_layers.append(conv_class(hidden_channels_list[i - 1], hidden_channels_list[i]))

        self.lin1 = Linear(hidden_channels_list[-1], hidden_channels_list[-1])
        self.lin2 = Linear(hidden_channels_list[-1], out_channels)

        self.SiLU = nn.SiLU()

    def forward(self, in_data):
        x, adj, mask = in_data.x, in_data.adj, in_data.mask

        for c in self.conv_layers:
            x = self.SiLU(c(x, adj, mask))

        x = x.mean(dim=1)
        x = self.SiLU(self.lin1(x))
        return self.lin2(x)
