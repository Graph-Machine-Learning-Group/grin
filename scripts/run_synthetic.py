import copy
import datetime
import os
import pathlib
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR

from lib import fillers, config
from lib.datasets import ChargedParticles
from lib.nn import models
from lib.nn.utils.metric_base import MaskedMetric
from lib.nn.utils.metrics import MaskedMAE, MaskedMAPE, MaskedMSE, MaskedMRE
from lib.utils import parser_utils
from lib.utils.parser_utils import str_to_bool


def has_graph_support(model_cls):
    return model_cls is models.GRINet


def get_model_classes(model_str):
    if model_str == 'brits':
        model, filler = models.BRITSNet, fillers.BRITSFiller
    elif model_str == 'grin':
        model, filler = models.GRINet, fillers.GraphFiller
    else:
        raise ValueError(f'Model {model_str} not available.')
    return model, filler


def parse_args():
    # Argument parser
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument("--model-name", type=str, default='bigrill')
    parser.add_argument("--config", type=str, default=None)
    # Dataset params
    parser.add_argument('--static-adj', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--window', type=int, default=50)
    parser.add_argument('--p-block', type=float, default=0.025)
    parser.add_argument('--p-point', type=float, default=0.025)
    parser.add_argument('--min-seq', type=int, default=5)
    parser.add_argument('--max-seq', type=int, default=10)
    parser.add_argument('--use-exogenous', type=str_to_bool, nargs='?', const=True, default=True)
    # Splitting/aggregation params
    parser.add_argument('--val-len', type=float, default=0.1)
    parser.add_argument('--test-len', type=float, default=0.2)
    # Training params
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--l2-reg', type=float, default=0.)
    parser.add_argument('--scaled-target', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--grad-clip-val', type=float, default=5.)
    parser.add_argument('--grad-clip-algorithm', type=str, default='norm')
    parser.add_argument('--loss-fn', type=str, default='mse_loss')
    parser.add_argument('--use-lr-schedule', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--whiten-prob', type=float, default=0.05)
    parser.add_argument('--pred-loss-weight', type=float, default=1.0)
    parser.add_argument('--warm-up', type=int, default=0)
    # graph params
    parser.add_argument("--adj-threshold", type=float, default=0.1)

    known_args, _ = parser.parse_known_args()
    model_cls, _ = get_model_classes(known_args.model_name)
    parser = model_cls.add_model_specific_args(parser)

    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as fp:
            config_args = yaml.load(fp, Loader=yaml.FullLoader)
        for arg in config_args:
            setattr(args, arg, config_args[arg])

    return args


def run_experiment(args):
    # Set configuration and seed
    args = copy.deepcopy(args)
    if args.seed < 0:
        args.seed = np.random.randint(1e9)
    torch.set_num_threads(1)
    pl.seed_everything(args.seed)

    ########################################
    # load dataset and model               #
    ########################################

    model_cls, filler_cls = get_model_classes(args.model_name)

    dataset = ChargedParticles(static_adj=args.static_adj,
                               window=args.window,
                               p_block=args.p_block,
                               p_point=args.p_point,
                               max_seq=args.max_seq,
                               min_seq=args.min_seq,
                               use_exogenous=args.use_exogenous,
                               graph_mode=has_graph_support(model_cls))

    dataset.split(args.val_len, args.test_len)

    # get adjacency matrix
    adj = dataset.get_similarity()
    np.fill_diagonal(adj, 0.)  # force adj with no self loop

    ########################################
    # create logdir and save configuration #
    ########################################

    exp_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.seed}"
    logdir = os.path.join(config['logs'], 'synthetic', args.model_name, exp_name)
    # save config for logging
    pathlib.Path(logdir).mkdir(parents=True)
    with open(os.path.join(logdir, 'config.yaml'), 'w') as fp:
        yaml.dump(parser_utils.config_dict_from_args(args), fp, indent=4, sort_keys=True)

    ########################################
    # predictor                            #
    ########################################

    # model's inputs
    if has_graph_support(model_cls):
        model_params = dict(adj=adj, d_in=dataset.n_channels, d_u=dataset.n_exogenous, n_nodes=dataset.n_nodes)
    else:
        model_params = dict(d_in=(dataset.n_channels * dataset.n_nodes), d_u=(dataset.n_channels * dataset.n_exogenous))
    model_kwargs = parser_utils.filter_args(args={**vars(args), **model_params},
                                            target_cls=model_cls,
                                            return_dict=True)

    # loss and metrics
    loss_fn = MaskedMetric(metric_fn=getattr(F, args.loss_fn),
                           compute_on_step=True,
                           metric_kwargs={'reduction': 'none'})

    metrics = {'mae': MaskedMAE(compute_on_step=False),
               'mape': MaskedMAPE(compute_on_step=False),
               'mse': MaskedMSE(compute_on_step=False),
               'mre': MaskedMRE(compute_on_step=False)}

    # filler's inputs
    scheduler_class = CosineAnnealingLR if args.use_lr_schedule else None
    additional_filler_hparams = dict(model_class=model_cls,
                                     model_kwargs=model_kwargs,
                                     optim_class=torch.optim.Adam,
                                     optim_kwargs={'lr': args.lr,
                                                   'weight_decay': args.l2_reg},
                                     loss_fn=loss_fn,
                                     metrics=metrics,
                                     scheduler_class=scheduler_class,
                                     scheduler_kwargs={
                                         'eta_min': 0.0001,
                                         'T_max': args.epochs
                                     },
                                     alpha=args.alpha,
                                     hint_rate=args.hint_rate,
                                     g_train_freq=args.g_train_freq,
                                     d_train_freq=args.d_train_freq)
    filler_kwargs = parser_utils.filter_args(args={**vars(args), **additional_filler_hparams},
                                             target_cls=filler_cls,
                                             return_dict=True)
    filler = filler_cls(**filler_kwargs)

    ########################################
    # logging options                      #
    ########################################

    # log number of parameters
    args.trainable_parameters = filler.trainable_parameters

    # log statistics on masks
    for mask_type in ['mask', 'eval_mask', 'training_mask']:
        mask_type_mean = getattr(dataset, mask_type).float().mean().item()
        setattr(args, mask_type, mask_type_mean)

    print(args)

    ########################################
    # training                             #
    ########################################

    # callbacks
    early_stop_callback = EarlyStopping(monitor='val_mse', patience=args.patience, mode='min')
    checkpoint_callback = ModelCheckpoint(dirpath=logdir, save_top_k=1, monitor='val_mse', mode='min')

    logger = TensorBoardLogger(logdir, name="model")

    trainer = pl.Trainer(max_epochs=args.epochs,
                         default_root_dir=logdir,
                         logger=logger,
                         gpus=1 if torch.cuda.is_available() else None,
                         gradient_clip_val=args.grad_clip_val,
                         gradient_clip_algorithm=args.grad_clip_algorithm,
                         callbacks=[early_stop_callback, checkpoint_callback])

    trainer.fit(filler,
                train_dataloader=dataset.train_dataloader(batch_size=args.batch_size),
                val_dataloaders=dataset.val_dataloader(batch_size=args.batch_size))

    ########################################
    # testing                              #
    ########################################

    filler.load_state_dict(torch.load(checkpoint_callback.best_model_path,
                                      lambda storage, loc: storage)['state_dict'])
    filler.freeze()
    trainer.test(filler, test_dataloaders=dataset.test_dataloader(batch_size=args.batch_size))
    filler.eval()


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)
