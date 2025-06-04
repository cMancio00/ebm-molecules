import torch
from torch import nn
from torch_geometric.nn.dense import DenseGCNConv
from torch_geometric.typing import OptTensor
import torch.nn.functional as F


class MultiEdgeDenseGCNConv(nn.Module):
    r"""
        DenseGCNConv with accept multi-dimensional adjacency matrix, i.e. A has shape N x N x F, but all the values are
        between 0 and 1
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_edge_types: int = 1,
            concat_edge_types: bool = False,
            bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_edge_types = num_edge_types
        self.concat_edge_types = concat_edge_types
        self.W = nn.Linear(self.in_channels, out_channels*num_edge_types, bias=False)
        self.W0 = nn.Linear(self.in_channels, out_channels*num_edge_types, bias=bias)

        if self.concat_edge_types:
            self.out_channels = out_channels*num_edge_types
        else:
            self.out_channels = out_channels

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.W.reset_parameters()
        self.W0.reset_parameters()

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: OptTensor = None) -> torch.Tensor:
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
        if x.dim() == 2:
            # add batch dimension
            x = x.unsqueeze(0)
            adj = adj.unsqueeze(0)

        if adj.dim() == 3:
            # add edge type dimension
            adj = adj.unsqueeze(-1)

        B, N, _, F = adj.size()  # has shape B x N x N x F

        # normalize adj
        deg_inv_sqrt = adj.sum(dim=2).clamp(min=1).pow(-0.5)  # has shape B x N x F
        adj = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(2)  # has shape B x N x N x F

        to_aggregate = self.W(x)  # has shae B x N x F*out_channels

        out = (adj.unsqueeze(-1) * to_aggregate.view(B, 1, N, F, -1)).sum(dim=2)  # has shape B x N x F x out_channels
        out = out + self.W0(x).view(B, N, F, -1)  # add self-loops

        if self.concat_edge_types:
            out = out.view(B, N, -1)  # concatenate over edge_tpyes
        else:
            out = out.sum(dim=2)  # sum over edge_tpyes

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out


class DenseGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels_list, out_channels, num_edge_types=1, concat_edge_types=False):
        super().__init__()

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(MultiEdgeDenseGCNConv(in_channels, hidden_channels_list[0],
                                                      num_edge_types, concat_edge_types))
        for i in range(1, len(hidden_channels_list)):
            self.conv_layers.append(MultiEdgeDenseGCNConv(self.conv_layers[-1].out_channels,
                                                          hidden_channels_list[i],
                                                          num_edge_types, concat_edge_types))

        last_out = self.conv_layers[-1].out_channels
        self.lin1 = nn.Linear(last_out, last_out)
        self.lin2 = nn.Linear(last_out, out_channels)

    def forward(self, in_data):
        x, adj, mask = in_data.x, in_data.adj, in_data.mask

        for c in self.conv_layers:
            x = F.silu(c(x, adj, mask))

        x = x.mean(dim=1)
        x = F.silu(self.lin1(x))
        return self.lin2(x)
