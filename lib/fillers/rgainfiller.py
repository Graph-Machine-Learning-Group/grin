import torch
from torch.nn import functional as F

from .multi_imputation_filler import MultiImputationFiller
from ..nn.utils.metric_base import MaskedMetric


class MaskedBCEWithLogits(MaskedMetric):
    def __init__(self,
                 mask_nans=False,
                 mask_inf=False,
                 compute_on_step=True,
                 dist_sync_on_step=False,
                 process_group=None,
                 dist_sync_fn=None,
                 at=None):
        super(MaskedBCEWithLogits, self).__init__(metric_fn=F.binary_cross_entropy_with_logits,
                                                  mask_nans=mask_nans,
                                                  mask_inf=mask_inf,
                                                  compute_on_step=compute_on_step,
                                                  dist_sync_on_step=dist_sync_on_step,
                                                  process_group=process_group,
                                                  dist_sync_fn=dist_sync_fn,
                                                  metric_kwargs={'reduction': 'none'},
                                                  at=at)


class RGAINFiller(MultiImputationFiller):
    def __init__(self,
                 model_class,
                 model_kwargs,
                 optim_class,
                 optim_kwargs,
                 loss_fn,
                 g_train_freq=1,
                 d_train_freq=5,
                 consistency_loss=False,
                 scaled_target=True,
                 whiten_prob=0.05,
                 hint_rate=0.7,
                 alpha=10.,
                 metrics=None,
                 scheduler_class=None,
                 scheduler_kwargs=None):
        super(RGAINFiller, self).__init__(model_class=model_class,
                                          model_kwargs=model_kwargs,
                                          optim_class=optim_class,
                                          optim_kwargs=optim_kwargs,
                                          loss_fn=loss_fn,
                                          scaled_target=scaled_target,
                                          whiten_prob=whiten_prob,
                                          metrics=metrics,
                                          consistency_loss=consistency_loss,
                                          scheduler_class=scheduler_class,
                                          scheduler_kwargs=scheduler_kwargs)
        # discriminator training params
        self.alpha = alpha
        self.g_train_freq = g_train_freq
        self.d_train_freq = d_train_freq
        self.masked_bce_loss = MaskedBCEWithLogits(compute_on_step=True)
        # activate manual optimization
        self.automatic_optimization = False
        self.hint_rate = hint_rate

    def training_step(self, batch, batch_idx):
        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)
        g_opt, d_opt = self.optimizers()
        schedulers = self.lr_schedulers()

        # Extract mask and target
        x = batch_data.pop('x')
        mask = batch_data['mask'].clone().detach()
        training_mask = torch.bernoulli(mask.clone().detach().float() * self.keep_prob).byte()
        eval_mask = batch_data.pop('eval_mask', None)
        y = batch_data.pop('y')

        ##########################
        #  generate imputations
        ##########################

        imputations = self.model.generator(x, training_mask)
        imputed_seq = imputations[0]
        target = self._preprocess(y, batch_preprocessing)
        y_hat = self._postprocess(imputed_seq, batch_preprocessing)

        x_in = training_mask * x + (1 - training_mask) * imputed_seq
        hint = torch.rand_like(training_mask, dtype=torch.float) < self.hint_rate
        hint = hint.byte()
        hint = hint * training_mask + (1 - hint) * 0.5

        #########################
        #  train generator
        #########################
        if (batch_idx % self.g_train_freq) == 0:

            g_opt.zero_grad()

            rec_loss = sum([torch.sqrt(self.loss_fn(imp, target, mask)) for imp in imputations])
            if self.consistency_loss:
                rec_loss += self._consistency_loss(imputations, mask)

            logits = self.model.discriminator(x_in, hint)
            # maximize logit
            adv_loss = self.masked_bce_loss(logits, torch.ones_like(logits), 1 - training_mask)

            g_loss = self.alpha * rec_loss + adv_loss

            self.manual_backward(g_loss)
            g_opt.step()

            # Logging
            metrics_mask = (mask | eval_mask) - training_mask
            self.train_metrics.update(y_hat.detach(), y, metrics_mask)  # all unseen data
            self.log_dict(self.train_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
            self.log('gen_loss', adv_loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
            self.log('imp_loss', rec_loss.detach(), on_step=True, on_epoch=True, logger=True, prog_bar=True)

        ###########################
        # train discriminator
        ###########################

        if (batch_idx % self.d_train_freq) == 0:
            d_opt.zero_grad()

            logits = self.model.discriminator(x_in.detach(), hint)
            d_loss = self.masked_bce_loss(logits, training_mask.to(logits.dtype))

            self.manual_backward(d_loss)
            d_opt.step()
            self.log('d_loss', d_loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)

        if (schedulers is not None) and self.trainer.is_last_batch:
            for sch in schedulers:
                sch.step()

    def configure_optimizers(self):
        opt_g = self.optim_class(self.model.generator.parameters(), **self.optim_kwargs)
        opt_d = self.optim_class(self.model.discriminator.parameters(), **self.optim_kwargs)
        optimizers = [opt_g, opt_d]
        if self.scheduler_class is not None:
            metric = self.scheduler_kwargs.pop('monitor', None)
            schedulers = [{"scheduler": self.scheduler_class(opt, **self.scheduler_kwargs), "monitor": metric}
                          for opt in optimizers]
            return optimizers, schedulers
        return optimizers
