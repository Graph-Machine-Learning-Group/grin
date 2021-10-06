import torch
from torch import nn

from ..utils.ops import reverse_tensor


class RNNImputer(nn.Module):
    """Fill the blanks with a 1-step-ahead GRU predictor."""

    def __init__(self, d_in, d_model, concat_mask=True, detach_inputs=False, state_init='zero', d_u=0):
        super(RNNImputer, self).__init__()
        self.concat_mask = concat_mask
        self.detach_inputs = detach_inputs
        self.state_init = state_init
        self.d_model = d_model
        self.input_dim = d_in + d_u if not concat_mask else 2 * d_in + d_u
        self.rnn_cell = nn.GRUCell(self.input_dim, d_model)
        self.read_out = nn.Linear(d_model, d_in)

    def init_hidden_state(self, x):
        if self.state_init == 'zero':
            return torch.zeros((x.size(0), self.d_model), device=x.device, dtype=x.dtype)
        if self.state_init == 'noise':
            return torch.randn(x.size(0), self.d_model, device=x.device, dtype=x.dtype)

    def _preprocess_input(self, x, x_hat, m, u):
        if self.detach_inputs:
            x_p = torch.where(m, x, x_hat.detach())
        else:
            x_p = torch.where(m, x, x_hat)

        if u is not None:
            x_p = torch.cat([x_p, u], -1)
        if self.concat_mask:
            x_p = torch.cat([x_p, m], -1)
        return x_p

    def forward(self, x, mask, u=None, return_hidden=False):
        # x: [batches, steps, features]
        steps = x.size(1)
        # ensure masked values are not visible
        x = torch.where(mask, x, torch.zeros_like(x))

        h = self.init_hidden_state(x)
        x_hat = self.read_out(h)
        hs = [h]
        preds = [x_hat]
        for s in range(steps - 1):
            u_t = None if u is None else u[:, s]
            x_t = self._preprocess_input(x[:, s], x_hat, mask[:, s], u_t)
            h = self.rnn_cell(x_t, h)
            x_hat = self.read_out(h)
            hs.append(h)
            preds.append(x_hat)

        x_hat = torch.stack(preds, 1)
        h = torch.stack(hs, 1)
        if return_hidden:
            return x_hat, h
        return x_hat

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--d-in', type=int)
        parser.add_argument('--d-model', type=int, default=None)
        return parser


class BiRNNImputer(nn.Module):
    """Fill the blanks with a 1-step-ahead GRU predictor."""

    def __init__(self, d_in, d_model, dropout=0., concat_mask=True, detach_inputs=False, state_init='zero', d_u=0):
        super(BiRNNImputer, self).__init__()
        self.d_model = d_model
        self.fwd_rnn = RNNImputer(d_in, d_model, concat_mask, detach_inputs=detach_inputs, state_init=state_init,
                                  d_u=d_u)
        self.bwd_rnn = RNNImputer(d_in, d_model, concat_mask, detach_inputs=detach_inputs, state_init=state_init,
                                  d_u=d_u)
        self.dropout = nn.Dropout(dropout)
        self.read_out = nn.Linear(2 * d_model, d_in)

    def forward(self, x, mask, u=None, return_hidden=False):
        # x: [batches, steps, features]
        x_hat_fwd, h_fwd = self.fwd_rnn(x, mask, u=u, return_hidden=True)
        x_hat_bwd, h_bwd = self.bwd_rnn(reverse_tensor(x, 1),
                                        reverse_tensor(mask, 1),
                                        u=reverse_tensor(u, 1) if u is not None else None,
                                        return_hidden=True)
        x_hat_bwd = reverse_tensor(x_hat_bwd, 1)
        h_bwd = reverse_tensor(h_bwd, 1)
        h = self.dropout(torch.cat([h_fwd, h_bwd], -1))
        x_hat = self.read_out(h)
        if return_hidden:
            return (x_hat, x_hat_fwd, x_hat_bwd), h
        return x_hat, x_hat_fwd, x_hat_bwd

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--d-in', type=int)
        parser.add_argument('--d-model', type=int, default=None)
        parser.add_argument('--dropout', type=float, default=0.)
        return parser
