import os

import numpy as np
import pandas as pd

from lib import datasets_path
from .pd_dataset import PandasDataset
from ..utils import sample_mask


class PemsBay(PandasDataset):
    def __init__(self):
        df, dist, mask = self.load()
        self.dist = dist
        super().__init__(dataframe=df, u=None, mask=mask, name='bay', freq='5T', aggr='nearest')

    def load(self, impute_zeros=True):
        path = os.path.join(datasets_path['bay'], 'pems_bay.h5')
        df = pd.read_hdf(path)
        datetime_idx = sorted(df.index)
        date_range = pd.date_range(datetime_idx[0], datetime_idx[-1], freq='5T')
        df = df.reindex(index=date_range)
        mask = ~np.isnan(df.values)
        df.fillna(method='ffill', axis=0, inplace=True)
        dist = self.load_distance_matrix(list(df.columns))
        return df.astype('float32'), dist, mask.astype('uint8')

    def load_distance_matrix(self, ids):
        path = os.path.join(datasets_path['bay'], 'pems_bay_dist.npy')
        try:
            dist = np.load(path)
        except:
            distances = pd.read_csv(os.path.join(datasets_path['bay'], 'distances_bay.csv'))
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

    def get_similarity(self, type='dcrnn', thr=0.1, force_symmetric=False, sparse=False):
        """
        Return similarity matrix among nodes. Implemented to match DCRNN.

        :param type: type of similarity matrix.
        :param thr: threshold to increase saprseness.
        :param trainlen: number of steps that can be used for computing the similarity.
        :param force_symmetric: force the result to be simmetric.
        :return: and NxN array representig similarity among nodes.
        """
        if type == 'dcrnn':
            finite_dist = self.dist.reshape(-1)
            finite_dist = finite_dist[~np.isinf(finite_dist)]
            sigma = finite_dist.std()
            adj = np.exp(-np.square(self.dist / sigma))
        elif type == 'stcn':
            sigma = 10
            adj = np.exp(-np.square(self.dist) / sigma)
        else:
            raise NotImplementedError
        adj[adj < thr] = 0.
        if force_symmetric:
            adj = np.maximum.reduce([adj, adj.T])
        if sparse:
            import scipy.sparse as sps
            adj = sps.coo_matrix(adj)
        return adj

    @property
    def mask(self):
        if self._mask is None:
            return self.df.values != 0.
        return self._mask


class MissingValuesPemsBay(PemsBay):
    SEED = 56789

    def __init__(self, p_fault=0.0015, p_noise=0.05):
        super(MissingValuesPemsBay, self).__init__()
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
