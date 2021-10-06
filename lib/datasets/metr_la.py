import os

import numpy as np
import pandas as pd

from lib import datasets_path
from .pd_dataset import PandasDataset
from ..utils import sample_mask


class MetrLA(PandasDataset):
    def __init__(self, impute_zeros=False, freq='5T'):

        df, dist, mask = self.load(impute_zeros=impute_zeros)
        self.dist = dist
        super().__init__(dataframe=df, u=None, mask=mask, name='la', freq=freq, aggr='nearest')

    def load(self, impute_zeros=True):
        path = os.path.join(datasets_path['la'], 'metr_la.h5')
        df = pd.read_hdf(path)
        datetime_idx = sorted(df.index)
        date_range = pd.date_range(datetime_idx[0], datetime_idx[-1], freq='5T')
        df = df.reindex(index=date_range)
        mask = ~np.isnan(df.values)
        if impute_zeros:
            mask = mask * (df.values != 0.).astype('uint8')
            df = df.replace(to_replace=0., method='ffill')
        else:
            mask = None
        dist = self.load_distance_matrix()
        return df, dist, mask

    def load_distance_matrix(self):
        path = os.path.join(datasets_path['la'], 'metr_la_dist.npy')
        try:
            dist = np.load(path)
        except:
            distances = pd.read_csv(os.path.join(datasets_path['la'], 'distances_la.csv'))
            with open(os.path.join(datasets_path['la'], 'sensor_ids_la.txt')) as f:
                ids = f.read().strip().split(',')
            num_sensors = len(ids)
            dist = np.ones((num_sensors, num_sensors), dtype=np.float32) * np.inf
            # Builds sensor id to index map.
            sensor_id_to_ind = {int(sensor_id): i for i, sensor_id in enumerate(ids)}

            # Fills cells in the matrix with distances.
            for row in distances.values:
                if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
                    continue
                dist[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]
            np.save(path, dist)
        return dist

    def get_similarity(self, thr=0.1, force_symmetric=False, sparse=False):
        finite_dist = self.dist.reshape(-1)
        finite_dist = finite_dist[~np.isinf(finite_dist)]
        sigma = finite_dist.std()
        adj = np.exp(-np.square(self.dist / sigma))
        adj[adj < thr] = 0.
        if force_symmetric:
            adj = np.maximum.reduce([adj, adj.T])
        if sparse:
            import scipy.sparse as sps
            adj = sps.coo_matrix(adj)
        return adj

    @property
    def mask(self):
        return self._mask


class MissingValuesMetrLA(MetrLA):
    SEED = 9101112

    def __init__(self, p_fault=0.0015, p_noise=0.05):
        super(MissingValuesMetrLA, self).__init__(impute_zeros=True)
        self.rng = np.random.default_rng(self.SEED)
        self.p_fault = p_fault
        self.p_noise = p_noise
        eval_mask = sample_mask(self.numpy().shape,
                                p=p_fault,
                                p_noise=p_noise,
                                min_seq=12,
                                max_seq=12 * 4,
                                rng=self.rng)
        self.eval_mask = (eval_mask & self.mask).astype('uint8')

    @property
    def training_mask(self):
        return self.mask if self.eval_mask is None else (self.mask & (1 - self.eval_mask))

    def splitter(self, dataset, val_len=0, test_len=0, window=0):
        idx = np.arange(len(dataset))
        if test_len < 1:
            test_len = int(test_len * len(idx))
        if val_len < 1:
            val_len = int(val_len * (len(idx) - test_len))
        test_start = len(idx) - test_len
        val_start = test_start - val_len
        return [idx[:val_start - window], idx[val_start:test_start - window], idx[test_start:]]