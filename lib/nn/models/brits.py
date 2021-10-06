import torch
from torch import nn

from ..layers import BRITS


class BRITSNet(nn.Module):
    def __init__(self,
                 d_in,
                 d_hidden=64):
        super(BRITSNet, self).__init__()
        self.birits = BRITS(input_size=d_in,
                            hidden_size=d_hidden)

    def forward(self, x, mask=None, **kwargs):
        # x: [batches, steps, features]
        imputations, predictions = self.birits(x, mask=mask)
        # predictions: [batch, directions, steps, features] x 3
        out = torch.mean(imputations, dim=1)  # -> [batch, steps, features]
        predictions = torch.cat(predictions, dim=1)  # -> [batch, directions * n_predictions, steps, features]
        # reshape
        imputations = torch.transpose(imputations, 0, 1)  # rearrange(imputations, 'b d s f -> d b s f')
        predictions = torch.transpose(predictions, 0, 1)  # rearrange(predictions, 'b d s f -> d b s f')
        return out, imputations, predictions

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--d-in', type=int)
        parser.add_argument('--d-hidden', type=int, default=64)
        return parser
