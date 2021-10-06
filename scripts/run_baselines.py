from argparse import ArgumentParser

import numpy as np
from fancyimpute import MatrixFactorization, IterativeImputer
from sklearn.neighbors import kneighbors_graph

from lib import datasets
from lib.utils import numpy_metrics
from lib.utils.parser_utils import str_to_bool

metrics = {
    'mae': numpy_metrics.masked_mae,
    'mse': numpy_metrics.masked_mse,
    'mre': numpy_metrics.masked_mre,
    'mape': numpy_metrics.masked_mape
}


def parse_args():
    parser = ArgumentParser()
    # experiment setting
    parser.add_argument('--datasets', nargs='+', type=str, default=['all'])
    parser.add_argument('--imputers', nargs='+', type=str, default=['all'])
    parser.add_argument('--n-runs', type=int, default=5)
    parser.add_argument('--in-sample', type=str_to_bool, nargs='?', const=True, default=True)
    # SpatialKNNImputer params
    parser.add_argument('--k', type=int, default=10)
    # MFImputer params
    parser.add_argument('--rank', type=int, default=10)
    # MICEImputer params
    parser.add_argument('--mice-iterations', type=int, default=100)
    parser.add_argument('--mice-n-features', type=int, default=None)
    args = parser.parse_args()
    # parse dataset
    if args.datasets[0] == 'all':
        args.datasets = ['air36', 'air', 'bay', 'irish', 'la', 'bay_noise', 'irish_noise', 'la_noise']
    # parse imputers
    if args.imputers[0] == 'all':
        args.imputers = ['mean', 'knn', 'mf', 'mice']
    if not args.in_sample:
        args.imputers = [name for name in args.imputers if name in ['mean', 'mice']]
    return args


class Imputer:
    short_name: str

    def __init__(self, method=None, is_deterministic=True, in_sample=True):
        self.name = self.__class__.__name__
        self.method = method
        self.is_deterministic = is_deterministic
        self.in_sample = in_sample

    def fit(self, x, mask):
        if not self.in_sample:
            x_hat = np.where(mask, x, np.nan)
            return self.method.fit(x_hat)

    def predict(self, x, mask):
        x_hat = np.where(mask, x, np.nan)
        if self.in_sample:
            return self.method.fit_transform(x_hat)
        else:
            return self.method.transform(x_hat)

    def params(self):
        return dict()


class SpatialKNNImputer(Imputer):
    short_name = 'knn'

    def __init__(self, adj, k=20):
        super(SpatialKNNImputer, self).__init__()
        self.k = k
        # normalize sim between [0, 1]
        sim = (adj + adj.min()) / (adj.max() + adj.min())
        knns = kneighbors_graph(1 - sim,
                                n_neighbors=self.k,
                                include_self=False,
                                metric='precomputed').toarray()
        self.knns = knns

    def fit(self, x, mask):
        pass

    def predict(self, x, mask):
        x = np.where(mask, x, 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            y_hat = (x @ self.knns.T) / (mask @ self.knns.T)
        y_hat[~np.isfinite(y_hat)] = x.mean()
        return np.where(mask, x, y_hat)

    def params(self):
        return dict(k=self.k)


class MeanImputer(Imputer):
    short_name = 'mean'

    def fit(self, x, mask):
        d = np.where(mask, x, np.nan)
        self.means = np.nanmean(d, axis=0, keepdims=True)

    def predict(self, x, mask):
        if self.in_sample:
            d = np.where(mask, x, np.nan)
            means = np.nanmean(d, axis=0, keepdims=True)
        else:
            means = self.means
        return np.where(mask, x, means)


class MatrixFactorizationImputer(Imputer):
    short_name = 'mf'

    def __init__(self, rank=10, loss='mae', verbose=0):
        method = MatrixFactorization(rank=rank, loss=loss, verbose=verbose)
        super(MatrixFactorizationImputer, self).__init__(method, is_deterministic=False, in_sample=True)

    def params(self):
        return dict(rank=self.method.rank)


class MICEImputer(Imputer):
    short_name = 'mice'

    def __init__(self, max_iter=100, n_nearest_features=None, in_sample=True, verbose=False):
        method = IterativeImputer(max_iter=max_iter, n_nearest_features=n_nearest_features, verbose=verbose)
        is_deterministic = n_nearest_features is None
        super(MICEImputer, self).__init__(method, is_deterministic=is_deterministic, in_sample=in_sample)

    def params(self):
        return dict(max_iter=self.method.max_iter, k=self.method.n_nearest_features or -1)


def get_dataset(dataset_name):
    if dataset_name[:3] == 'air':
        dataset = datasets.AirQuality(impute_nans=True, small=dataset_name[3:] == '36')
    elif dataset_name == 'bay':
        dataset = datasets.MissingValuesPemsBay()
    elif dataset_name == 'la':
        dataset = datasets.MissingValuesMetrLA()
    elif dataset_name == 'la_noise':
        dataset = datasets.MissingValuesMetrLA(p_fault=0., p_noise=0.25)
    elif dataset_name == 'bay_noise':
        dataset = datasets.MissingValuesPemsBay(p_fault=0., p_noise=0.25)
    else:
        raise ValueError(f"Dataset {dataset_name} not available in this setting.")

    # split in train/test
    if isinstance(dataset, datasets.AirQuality):
        test_slice = np.in1d(dataset.df.index.month, dataset.test_months)
        train_slice = ~test_slice
    else:
        train_slice = np.zeros(len(dataset)).astype(bool)
        train_slice[:-int(0.2 * len(dataset))] = True

    # integrate back eval values in dataset
    dataset.eval_mask[train_slice] = 0

    return dataset, train_slice


def get_imputer(imputer_name, args):
    if imputer_name == 'mean':
        imputer = MeanImputer(in_sample=args.in_sample)
    elif imputer_name == 'knn':
        imputer = SpatialKNNImputer(adj=args.adj, k=args.k)
    elif imputer_name == 'mf':
        imputer = MatrixFactorizationImputer(rank=args.rank)
    elif imputer_name == 'mice':
        imputer = MICEImputer(max_iter=args.mice_iterations,
                              n_nearest_features=args.mice_n_features,
                              in_sample=args.in_sample)
    else:
        raise ValueError(f"Imputer {imputer_name} not available in this setting.")
    return imputer


def run(imputer, dataset, train_slice):
    test_slice = ~train_slice
    if args.in_sample:
        x_train, mask_train = dataset.numpy(), dataset.training_mask
        y_hat = imputer.predict(x_train, mask_train)[test_slice]
    else:
        x_train, mask_train = dataset.numpy()[train_slice], dataset.training_mask[train_slice]
        imputer.fit(x_train, mask_train)
        x_test, mask_test = dataset.numpy()[test_slice], dataset.training_mask[test_slice]
        y_hat = imputer.predict(x_test, mask_test)

    # Evaluate model
    y_true = dataset.numpy()[test_slice]
    eval_mask = dataset.eval_mask[test_slice]

    for metric, metric_fn in metrics.items():
        error = metric_fn(y_hat, y_true, eval_mask)
        print(f'{imputer.name} on {ds_name} {metric}: {error:.4f}')


if __name__ == '__main__':

    args = parse_args()
    print(args.__dict__)

    for ds_name in args.datasets:

        dataset, train_slice = get_dataset(ds_name)
        args.adj = dataset.get_similarity(thr=0.1)

        # Instantiate imputers
        imputers = [get_imputer(name, args) for name in args.imputers]

        for imputer in imputers:
            n_runs = 1 if imputer.is_deterministic else args.n_runs
            for _ in range(n_runs):
                run(imputer, dataset, train_slice)
