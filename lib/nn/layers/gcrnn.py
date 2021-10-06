import torch
import torch.nn as nn

from .spatial_conv import SpatialConvOrderK


class GCGRUCell(nn.Module):
    """
    Graph Convolution Gated Recurrent Unit Cell.
    """

    def __init__(self, d_in, num_units, support_len, order, activation='tanh'):
        """
        :param num_units: the hidden dim of rnn
        :param support_len: the (weighted) adjacency matrix of the graph, in numpy ndarray form
        :param order: the max diffusion step
        :param activation: if None, don't do activation for cell state
        """
        super(GCGRUCell, self).__init__()
        self.activation_fn = getattr(torch, activation)

        self.forget_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len,
                                             order=order)
        self.update_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len,
                                             order=order)
        self.c_gate = SpatialConvOrderK(c_in=d_in + num_units, c_out=num_units, support_len=support_len, order=order)

    def forward(self, x, h, adj):
        """
        :param x: (B, input_dim, num_nodes)
        :param h: (B, num_units, num_nodes)
        :param adj: (num_nodes, num_nodes)
        :return:
        """
        # we start with bias 1.0 to not reset and not update
        x_gates = torch.cat([x, h], dim=1)
        r = torch.sigmoid(self.forget_gate(x_gates, adj))
        u = torch.sigmoid(self.update_gate(x_gates, adj))
        x_c = torch.cat([x, r * h], dim=1)
        c = self.c_gate(x_c, adj)  # batch_size, self._num_nodes * output_size
        c = self.activation_fn(c)
        return u * h + (1. - u) * c


class GCRNN(nn.Module):
    def __init__(self,
                 d_in,
                 d_model,
                 d_out,
                 n_layers,
                 support_len,
                 kernel_size=2):
        super(GCRNN, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.d_out = d_out
        self.n_layers = n_layers
        self.ks = kernel_size
        self.support_len = support_len
        self.rnn_cells = nn.ModuleList()
        for i in range(self.n_layers):
            self.rnn_cells.append(GCGRUCell(d_in=self.d_in if i == 0 else self.d_model,
                                            num_units=self.d_model, support_len=self.support_len, order=self.ks))
        self.output_layer = nn.Conv2d(self.d_model, self.d_out, kernel_size=1)

    def init_hidden_states(self, x):
        return [torch.zeros(size=(x.shape[0], self.d_model, x.shape[2])).to(x.device) for _ in range(self.n_layers)]

    def single_pass(self, x, h, adj):
        out = x
        for l, layer in enumerate(self.rnn_cells):
            out = h[l] = layer(out, h[l], adj)
        return out, h

    def forward(self, x, adj, h=None):
        # x:[batch, features, nodes, steps]
        *_, steps = x.size()
        if h is None:
            h = self.init_hidden_states(x)
        # temporal conv
        for step in range(steps):
            out, h = self.single_pass(x[..., step], h, adj)

        return self.output_layer(out[..., None])
