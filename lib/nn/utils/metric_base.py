from functools import partial

import torch
from pytorch_lightning.metrics import Metric
from torchmetrics.utilities.checks import _check_same_shape


class MaskedMetric(Metric):
    def __init__(self,
                 metric_fn,
                 mask_nans=False,
                 mask_inf=False,
                 compute_on_step=True,
                 dist_sync_on_step=False,
                 process_group=None,
                 dist_sync_fn=None,
                 metric_kwargs=None,
                 at=None):
        super(MaskedMetric, self).__init__(compute_on_step=compute_on_step,
                                           dist_sync_on_step=dist_sync_on_step,
                                           process_group=process_group,
                                           dist_sync_fn=dist_sync_fn)

        if metric_kwargs is None:
            metric_kwargs = dict()
        self.metric_fn = partial(metric_fn, **metric_kwargs)
        self.mask_nans = mask_nans
        self.mask_inf = mask_inf
        if at is None:
            self.at = slice(None)
        else:
            self.at = slice(at, at + 1)
        self.add_state('value', dist_reduce_fx='sum', default=torch.tensor(0.).float())
        self.add_state('numel', dist_reduce_fx='sum', default=torch.tensor(0))

    def _check_mask(self, mask, val):
        if mask is None:
            mask = torch.ones_like(val).byte()
        else:
            _check_same_shape(mask, val)
        if self.mask_nans:
            mask = mask * ~torch.isnan(val)
        if self.mask_inf:
            mask = mask * ~torch.isinf(val)
        return mask

    def _compute_masked(self, y_hat, y, mask):
        _check_same_shape(y_hat, y)
        val = self.metric_fn(y_hat, y)
        mask = self._check_mask(mask, val)
        val = torch.where(mask, val, torch.tensor(0., device=val.device).float())
        return val.sum(), mask.sum()

    def _compute_std(self, y_hat, y):
        _check_same_shape(y_hat, y)
        val = self.metric_fn(y_hat, y)
        return val.sum(), val.numel()

    def is_masked(self, mask):
        return self.mask_inf or self.mask_nans or (mask is not None)

    def update(self, y_hat, y, mask=None):
        y_hat = y_hat[:, self.at]
        y = y[:, self.at]
        if mask is not None:
            mask = mask[:, self.at]
        if self.is_masked(mask):
            val, numel = self._compute_masked(y_hat, y, mask)
        else:
            val, numel = self._compute_std(y_hat, y)
        self.value += val
        self.numel += numel

    def compute(self):
        if self.numel > 0:
            return self.value / self.numel
        return self.value
