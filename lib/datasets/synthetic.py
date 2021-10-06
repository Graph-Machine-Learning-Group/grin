import os.path

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset, DataLoader, Subset

from lib import datasets_path


def generate_mask(shape, p_block=0.01, p_point=0.01, max_seq=1, min_seq=1, rng=None):
    """Generate mask in which 1 denotes valid values, 0 missing ones. Assuming shape=(steps, ...)."""
    if rng is None:
        rand = np.random.random
        randint = np.random.randint
    else:
        rand = rng.random
        randint = rng.integers
    # init mask
    mask = np.ones(shape, dtype='uint8')
    # block missing
    if p_block > 0:
        assert max_seq >= min_seq
        for col in range(shape[1]):
            i = 0
            while i < shape[0]:
                if rand() > p_block:
                    i += 1
                else:
                    fault_len = int(randint(min_seq, max_seq + 1))
                    mask[i:i + fault_len, col] = 0
                    i += fault_len + 1  # at least one valid value between two blocks
    # point missing
    # let values before and after block missing always valid
    diff = np.zeros(mask.shape, dtype='uint8')
    diff[:-1] |= np.diff(mask, axis=0) < 0
    diff[1:] |= np.diff(mask, axis=0) > 0
    mask = np.where(mask - diff, rand(shape) > p_point, mask)
    return mask


class SyntheticDataset(Dataset):
    SEED: int

    def __init__(self, filename,
                 window=None,
                 p_block=0.05,
                 p_point=0.05,
                 max_seq=6,
                 min_seq=4,
                 use_exogenous=True,
                 mask_exogenous=True,
                 graph_mode=True):
        super(SyntheticDataset, self).__init__()
        self.mask_exogenous = mask_exogenous
        self.use_exogenous = use_exogenous
        self.graph_mode = graph_mode
        # fetch data
        content = self.load(filename)
        self.window = window if window is not None else content['loc'].shape[1]
        self.loc = torch.tensor(content['loc'][:, :self.window]).float()
        self.vel = torch.tensor(content['vel'][:, :self.window]).float()
        self.adj = content['adj']
        self.SEED = content['seed'].item()
        # compute masks
        self.rng = np.random.default_rng(self.SEED)
        mask_shape = (len(self), self.window, self.n_nodes, 1)
        mask = generate_mask(mask_shape,
                             p_block=p_block,
                             p_point=p_point,
                             max_seq=max_seq,
                             min_seq=min_seq,
                             rng=self.rng).repeat(self.n_channels, -1)
        eval_mask = 1 - generate_mask(mask_shape,
                                      p_block=p_block,
                                      p_point=p_point,
                                      max_seq=max_seq,
                                      min_seq=min_seq,
                                      rng=self.rng).repeat(self.n_channels, -1)
        self.mask = torch.tensor(mask).byte()
        self.eval_mask = torch.tensor(eval_mask).byte() & self.mask
        # store splitting indices
        self.train_idxs = None
        self.val_idxs = None
        self.test_idxs = None

    def __len__(self):
        return self.loc.size(0)

    def __getitem__(self, index):
        eval_mask = self.eval_mask[index]
        mask = self.training_mask[index]
        x = mask * self.loc[index]
        res = dict(x=x, mask=mask, eval_mask=eval_mask)
        if self.use_exogenous:
            u = self.vel[index]
            if self.mask_exogenous:
                u *= mask.all(-1, keepdims=True)
            res.update(u=u)
        res.update(y=self.loc[index])
        if not self.graph_mode:
            res = {k: rearrange(v, 's n f -> s (n f)') for k, v in res.items()}
        return res

    @property
    def n_channels(self):
        return self.loc.size(-1)

    @property
    def n_nodes(self):
        return self.loc.size(-2)

    @property
    def n_exogenous(self):
        return self.vel.size(-1) if self.use_exogenous else 0

    @property
    def training_mask(self):
        return self.mask if self.eval_mask is None else (self.mask & (1 - self.eval_mask))

    @staticmethod
    def load(filename):
        return np.load(filename)

    def get_similarity(self, sparse=False):
        return self.adj

    # Splitting options

    def split(self, val_len=0, test_len=0):
        idx = np.arange(len(self))
        if test_len < 1:
            test_len = int(test_len * len(idx))
        if val_len < 1:
            val_len = int(val_len * (len(idx) - test_len))
        test_start = len(idx) - test_len
        val_start = test_start - val_len
        # split dataset
        self.train_idxs = idx[:val_start]
        self.val_idxs = idx[val_start:test_start]
        self.test_idxs = idx[test_start:]

    def train_dataloader(self, shuffle=True, batch_size=32):
        return DataLoader(Subset(self, self.train_idxs), shuffle=shuffle, batch_size=batch_size, drop_last=True)

    def val_dataloader(self, shuffle=False, batch_size=32):
        return DataLoader(Subset(self, self.val_idxs), shuffle=shuffle, batch_size=batch_size)

    def test_dataloader(self, shuffle=False, batch_size=32):
        return DataLoader(Subset(self, self.test_idxs), shuffle=shuffle, batch_size=batch_size)


class ChargedParticles(SyntheticDataset):

    def __init__(self, static_adj=False,
                 window=None,
                 p_block=0.05,
                 p_point=0.05,
                 max_seq=6,
                 min_seq=4,
                 use_exogenous=True,
                 mask_exogenous=True,
                 graph_mode=True):
        if static_adj:
            filename = os.path.join(datasets_path['synthetic'], 'charged_static.npz')
        else:
            filename = os.path.join(datasets_path['synthetic'], 'charged_varying.npz')
        self.static_adj = static_adj
        super(ChargedParticles, self).__init__(filename, window,
                                               p_block=p_block,
                                               p_point=p_point,
                                               max_seq=max_seq,
                                               min_seq=min_seq,
                                               use_exogenous=use_exogenous,
                                               mask_exogenous=mask_exogenous,
                                               graph_mode=graph_mode)
        charges = self.load(filename)['charges']
        self.charges = torch.tensor(charges).float()

    def __getitem__(self, item):
        res = super(ChargedParticles, self).__getitem__(item)
        # add charges as exogenous features
        if self.use_exogenous:
            charges = self.charges[item] if not self.static_adj else self.charges
            stacked_charges = charges[None].expand(self.window, -1, -1)
            if not self.graph_mode:
                stacked_charges = rearrange(stacked_charges, 's n f -> s (n f)')
            res.update(u=torch.cat([res['u'], stacked_charges], -1))
        return res

    def get_similarity(self, sparse=False):
        return np.ones((self.n_nodes, self.n_nodes)) - np.eye(self.n_nodes)

    @property
    def n_exogenous(self):
        if self.use_exogenous:
            return super(ChargedParticles, self).n_exogenous + 1  # add charges to features
        return 0
