import torch
from einops import rearrange
from torch import nn

from ..layers import MPGRUImputer, SpatialConvOrderK
from ..utils.ops import reverse_tensor
from ...utils.parser_utils import str_to_bool


class MPGRUNet(nn.Module):
    def __init__(self,
                 adj,
                 d_in,
                 d_hidden,
                 d_ff=0,
                 d_u=0,
                 n_layers=1,
                 dropout=0.,
                 kernel_size=2,
                 support_len=2,
                 layer_norm=False,
                 impute_only_holes=True):
        super(MPGRUNet, self).__init__()
        self.register_buffer('adj', torch.tensor(adj).float())
        n_nodes = adj.shape[0]
        self.gcgru = MPGRUImputer(input_size=d_in,
                                  hidden_size=d_hidden,
                                  ff_size=d_ff,
                                  u_size=d_u,
                                  n_layers=n_layers,
                                  dropout=dropout,
                                  kernel_size=kernel_size,
                                  support_len=support_len,
                                  layer_norm=layer_norm,
                                  n_nodes=n_nodes)
        self.impute_only_holes = impute_only_holes

    def forward(self, x, mask=None, u=None, h=None):
        # x: [batches, steps, nodes, channels] -> [batches, channels, nodes, steps]
        x = rearrange(x, 'b s n c -> b c n s')
        if mask is not None:
            mask = rearrange(mask, 'b s n c -> b c n s')
        if u is not None:
            u = rearrange(u, 'b s n c -> b c n s')

        adj = SpatialConvOrderK.compute_support(self.adj, x.device)
        imputation, _ = self.gcgru(x, adj, mask=mask, u=u, h=h)

        # In evaluation stage impute only missing values
        if self.impute_only_holes and not self.training:
            imputation = torch.where(mask, x, imputation)

        # out: [batches, channels, nodes, steps] -> [batches, steps, nodes, channels]
        imputation = rearrange(imputation, 'b c n s -> b s n c')

        return imputation

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--d-hidden', type=int, default=64)
        parser.add_argument('--d-ff', type=int, default=64)
        parser.add_argument('--n-layers', type=int, default=1)
        parser.add_argument('--kernel-size', type=int, default=2)
        parser.add_argument('--layer-norm', type=str_to_bool, nargs='?', const=True, default=False)
        parser.add_argument('--impute-only-holes', type=str_to_bool, nargs='?', const=True, default=True)
        parser.add_argument('--dropout', type=float, default=0.)
        return parser


class BiMPGRUNet(nn.Module):
    def __init__(self,
                 adj,
                 d_in,
                 d_hidden,
                 d_ff=0,
                 d_u=0,
                 n_layers=1,
                 dropout=0.,
                 kernel_size=2,
                 support_len=2,
                 layer_norm=False,
                 embedding_size=0,
                 merge='mlp',
                 impute_only_holes=True,
                 autoencoder_mode=False):
        super(BiMPGRUNet, self).__init__()
        self.register_buffer('adj', torch.tensor(adj).float())
        n_nodes = adj.shape[0]
        self.gcgru_fwd = MPGRUImputer(input_size=d_in,
                                      hidden_size=d_hidden,
                                      u_size=d_u,
                                      n_layers=n_layers,
                                      dropout=dropout,
                                      kernel_size=kernel_size,
                                      support_len=support_len,
                                      layer_norm=layer_norm,
                                      n_nodes=n_nodes,
                                      autoencoder_mode=autoencoder_mode)
        self.gcgru_bwd = MPGRUImputer(input_size=d_in,
                                      hidden_size=d_hidden,
                                      u_size=d_u,
                                      n_layers=n_layers,
                                      dropout=dropout,
                                      kernel_size=kernel_size,
                                      support_len=support_len,
                                      layer_norm=layer_norm,
                                      n_nodes=n_nodes,
                                      autoencoder_mode=autoencoder_mode)
        self.impute_only_holes = impute_only_holes

        if n_nodes is None:
            embedding_size = 0
        if embedding_size > 0:
            self.emb = nn.Parameter(torch.empty(embedding_size, n_nodes))
            nn.init.kaiming_normal_(self.emb, nonlinearity='relu')
        else:
            self.register_parameter('emb', None)

        if merge == 'mlp':
            self._impute_from_states = True
            self.out = nn.Sequential(
                nn.Conv2d(in_channels=2 * d_hidden + d_in + embedding_size,
                          out_channels=d_ff, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=d_ff, out_channels=d_in, kernel_size=1)
            )
        elif merge in ['mean', 'sum', 'min', 'max']:
            self._impute_from_states = False
            self.out = getattr(torch, merge)
        else:
            raise ValueError("Merge option %s not allowed." % merge)

    def forward(self, x, mask=None, u=None, h=None):
        # x: [batches, steps, nodes, channels] -> [batches, channels, nodes, steps]
        x = rearrange(x, 'b s n c -> b c n s')
        if mask is not None:
            mask = rearrange(mask, 'b s n c -> b c n s')
        if u is not None:
            u = rearrange(u, 'b s n c -> b c n s')

        adj = SpatialConvOrderK.compute_support(self.adj, x.device)

        # Forward
        fwd_pred, fwd_states = self.gcgru_fwd(x, adj, mask=mask, u=u)
        # Backward
        rev_x, rev_mask, rev_u = [reverse_tensor(tens, axis=-1) for tens in (x, mask, u)]
        bwd_res = self.gcgru_bwd(rev_x, adj, mask=rev_mask, u=rev_u)
        bwd_pred, bwd_states = [reverse_tensor(res, axis=-1) for res in bwd_res]

        if self._impute_from_states:
            inputs = [fwd_states[-1], bwd_states[-1], mask]  # take only state of last gcgru layer
            if self.emb is not None:
                b, *_, s = x.shape  # fwd_h: [batches, channels, nodes, steps]
                inputs += [self.emb.view(1, *self.emb.shape, 1).expand(b, -1, -1, s)]  # stack emb for batches and steps
            imputation = torch.cat(inputs, dim=1)
            imputation = self.out(imputation)
        else:
            imputation = torch.stack([fwd_pred, bwd_pred], dim=1)
            imputation = self.out(imputation, dim=1)

        # In evaluation stage impute only missing values
        if self.impute_only_holes and not self.training:
            imputation = torch.where(mask, x, imputation)

        # out: [batches, channels, nodes, steps] -> [batches, steps, nodes, channels]
        imputation = rearrange(imputation, 'b c n s -> b s n c')

        return imputation

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--d-hidden', type=int, default=64)
        parser.add_argument('--d-ff', type=int, default=64)
        parser.add_argument('--n-layers', type=int, default=1)
        parser.add_argument('--kernel-size', type=int, default=2)
        parser.add_argument('--d-emb', type=int, default=8)
        parser.add_argument('--layer-norm', type=str_to_bool, nargs='?', const=True, default=False)
        parser.add_argument('--merge', type=str, default='mlp')
        parser.add_argument('--impute-only-holes', type=str_to_bool, nargs='?', const=True, default=True)
        parser.add_argument('--dropout', type=float, default=0.)
        parser.add_argument('--autoencoder-mode', type=str_to_bool, nargs='?', const=True, default=False)
        return parser
