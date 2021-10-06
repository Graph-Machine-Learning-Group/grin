import torch
from pytorch_lightning.core.decorators import auto_move_data

from . import Filler


class MultiImputationFiller(Filler):
    """
    Filler with multiple imputation outputs
    """

    def __init__(self,
                 model_class,
                 model_kwargs,
                 optim_class,
                 optim_kwargs,
                 loss_fn,
                 consistency_loss=False,
                 scaled_target=False,
                 whiten_prob=0.05,
                 metrics=None,
                 scheduler_class=None,
                 scheduler_kwargs=None):

        super().__init__(model_class,
                         model_kwargs,
                         optim_class,
                         optim_kwargs,
                         loss_fn,
                         scaled_target,
                         whiten_prob,
                         metrics,
                         scheduler_class,
                         scheduler_kwargs)
        self.consistency_loss = consistency_loss

    @auto_move_data
    def forward(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        assert isinstance(out, (list, tuple))
        if self.training:
            return out
        return out[0]  # we assume that the final imputation is the first one

    def _consistency_loss(self, imputations, mask):
        from itertools import combinations
        return sum([self.loss_fn(imp1, imp2, mask) for imp1, imp2 in combinations(imputations, 2)])

    def training_step(self, batch, batch_idx):
        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        # Extract mask and target
        mask = batch_data['mask'].clone().detach()
        batch_data['mask'] = torch.bernoulli(mask.clone().detach().float() * self.keep_prob).byte()
        eval_mask = batch_data.pop('eval_mask', None)
        y = batch_data.pop('y')

        # Compute predictions and compute loss
        imputations = self.predict_batch(batch, preprocess=False, postprocess=False)

        if self.scaled_target:
            target = self._preprocess(y, batch_preprocessing)
        else:
            target = y
            imputations = [self._postprocess(imp, batch_preprocessing) for imp in imputations]

        loss = sum([self.loss_fn(imp, target, mask) for imp in imputations])
        if self.consistency_loss:
            loss += self._consistency_loss(imputations, mask)

        # Logging
        metrics_mask = (mask | eval_mask) - batch_data['mask']  # all unseen data

        x_hat = imputations[0]
        x_hat = self._postprocess(x_hat, batch_preprocessing)
        self.train_metrics.update(x_hat.detach(), y, metrics_mask)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('train_loss', loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
        return loss
