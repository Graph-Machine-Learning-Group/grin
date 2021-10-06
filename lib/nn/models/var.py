import torch
from einops import rearrange
from torch import nn

from lib import epsilon


class VAR(nn.Module):
    def __init__(self, order, d_in, d_out=None, steps_ahead=1, bias=True):
        super(VAR, self).__init__()
        self.order = order
        self.d_in = d_in
        self.d_out = d_out if d_out is not None else d_in
        self.steps_ahead = steps_ahead
        self.lin = nn.Linear(order * d_in, steps_ahead * self.d_out, bias=bias)

    def forward(self, x):
        # x: [batches, steps, features]
        x = rearrange(x, 'b s f -> b (s f)')
        out = self.lin(x)
        out = rearrange(out, 'b (s f) -> b s f', s=self.steps_ahead, f=self.d_out)
        return out

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--order', type=int)
        parser.add_argument('--d-in', type=int)
        parser.add_argument('--d-out', type=int, default=None)
        parser.add_argument('--steps-ahead', type=int, default=1)
        return parser


class VARImputer(nn.Module):
    """Fill the blanks with a 1-step-ahead VAR predictor."""

    def __init__(self, order, d_in, padding='mean'):
        super(VARImputer, self).__init__()
        assert padding in ['mean', 'zero']
        self.order = order
        self.padding = padding
        self.predictor = VAR(order, d_in, d_out=d_in, steps_ahead=1)

    def forward(self, x, mask=None):
        # x: [batches, steps, features]
        batch_size, steps, n_feats = x.shape
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.uint8)
        x = x * mask
        # pad input sequence to start filling from first step
        if self.padding == 'mean':
            mean = torch.sum(x, 1) / (torch.sum(mask, 1) + epsilon)
            pad = torch.repeat_interleave(mean.unsqueeze(1), self.order, 1)
        elif self.padding == 'zero':
            pad = torch.zeros((batch_size, self.order, n_feats)).to(x.device)
        x = torch.cat([pad, x], 1)
        # x: [batch, order + steps, features]
        x = [x[:, i] for i in range(x.shape[1])]
        for s in range(steps):
            x_hat = self.predictor(torch.stack(x[s:s + self.order], 1))
            x_hat = x_hat[:, 0]
            x[s + self.order] = torch.where(mask[:, s], x[s + self.order], x_hat)
        x = torch.stack(x[self.order:], 1)  # remove padding
        return x

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--order', type=int)
        parser.add_argument('--d-in', type=int)
        parser.add_argument("--padding", type=str, default='mean')
        return parser
