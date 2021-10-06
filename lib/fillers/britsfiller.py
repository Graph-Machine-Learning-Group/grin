import torch

from . import Filler
from ..nn import BRITS


class BRITSFiller(Filler):

    def training_step(self, batch, batch_idx):
        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        # Extract mask and target
        mask = batch_data['mask'].clone().detach()
        batch_data['mask'] = torch.bernoulli(mask.clone().detach().float() * self.keep_prob).byte()
        eval_mask = batch_data.pop('eval_mask', None)
        y = batch_data.pop('y')

        # Compute predictions and compute loss
        out, imputations, predictions = self.predict_batch(batch, preprocess=False, postprocess=False)

        if self.scaled_target:
            target = self._preprocess(y, batch_preprocessing)
        else:
            target = y
            imputations = [self._postprocess(imp, batch_preprocessing) for imp in imputations]
            predictions = [self._postprocess(prd, batch_preprocessing) for prd in predictions]

        loss = sum([self.loss_fn(pred, target, mask) for pred in predictions])
        loss += BRITS.consistency_loss(*imputations)

        # Logging
        metrics_mask = (mask | eval_mask) - batch_data['mask']  # all unseen data
        out = self._postprocess(out, batch_preprocessing)
        self.train_metrics.update(out.detach(), y, metrics_mask)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        # Extract mask and target
        mask = batch_data.get('mask')
        eval_mask = batch_data.pop('eval_mask', None)
        y = batch_data.pop('y')

        # Compute predictions and compute loss
        out, imputations, predictions = self.predict_batch(batch, preprocess=False, postprocess=False)

        if self.scaled_target:
            target = self._preprocess(y, batch_preprocessing)
        else:
            target = y
            predictions = [self._postprocess(prd, batch_preprocessing) for prd in predictions]

        val_loss = sum([self.loss_fn(pred, target, mask) for pred in predictions])

        # Logging
        out = self._postprocess(out, batch_preprocessing)
        self.val_metrics.update(out.detach(), y, eval_mask)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_loss', val_loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
        return val_loss

    def test_step(self, batch, batch_idx):
        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        # Extract mask and target
        eval_mask = batch_data.pop('eval_mask', None)
        y = batch_data.pop('y')

        # Compute outputs and rescale
        imputation, *_ = self.predict_batch(batch, preprocess=False, postprocess=True)
        test_loss = self.loss_fn(imputation, y, eval_mask)

        # Logging
        self.test_metrics.update(imputation.detach(), y, eval_mask)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('test_loss', test_loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
        return test_loss
