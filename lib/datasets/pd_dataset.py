import numpy as np
import pandas as pd
import torch


class PandasDataset:
    def __init__(self, dataframe: pd.DataFrame, u: pd.DataFrame = None, name='pd-dataset', mask=None, freq=None,
                 aggr='sum', **kwargs):
        """
        Initialize a tsl dataset from a pandas dataframe.


        :param dataframe: dataframe containing the data, shape: n_steps, n_nodes
        :param u: dataframe with exog variables
        :param name: optional name of the dataset
        :param mask: mask for valid data (1:valid, 0:not valid)
        :param freq: force a frequency (possibly by resampling)
        :param aggr: aggregation method after resampling
        """
        super().__init__()
        self.name = name

        # set dataset dataframe
        self.df = dataframe

        # set optional exog_variable dataframe
        # make sure to consider only the overlapping part of the two dataframes
        # assumption u.index \in df.index
        idx = sorted(self.df.index)
        self.start = idx[0]
        self.end = idx[-1]

        if u is not None:
            self.u = u[self.start:self.end]
        else:
            self.u = None

        if mask is not None:
            mask = np.asarray(mask).astype('uint8')
        self._mask = mask

        if freq is not None:
            self.resample_(freq=freq, aggr=aggr)
        else:
            self.freq = self.df.index.inferred_freq
            # make sure that all the dataframes are aligned
            self.resample_(self.freq, aggr=aggr)

        assert 'T' in self.freq
        self.samples_per_day = int(60 / int(self.freq[:-1]) * 24)

    def __repr__(self):
        return "{}(nodes={}, length={})".format(self.__class__.__name__, self.n_nodes, self.length)

    @property
    def has_mask(self):
        return self._mask is not None

    @property
    def has_u(self):
        return self.u is not None

    def resample_(self, freq, aggr):
        resampler = self.df.resample(freq)
        idx = self.df.index
        if aggr == 'sum':
            self.df = resampler.sum()
        elif aggr == 'mean':
            self.df = resampler.mean()
        elif aggr == 'nearest':
            self.df = resampler.nearest()
        else:
            raise ValueError(f'{aggr} if not a valid aggregation method.')

        if self.has_mask:
            resampler = pd.DataFrame(self._mask, index=idx).resample(freq)
            self._mask = resampler.min().to_numpy()

        if self.has_u:
            resampler = self.u.resample(freq)
            self.u = resampler.nearest()
        self.freq = freq

    def dataframe(self) -> pd.DataFrame:
        return self.df.copy()

    @property
    def length(self):
        return self.df.values.shape[0]

    @property
    def n_nodes(self):
        return self.df.values.shape[1]

    @property
    def mask(self):
        if self._mask is None:
            return np.ones_like(self.df.values).astype('uint8')
        return self._mask

    def numpy(self, return_idx=False):
        if return_idx:
            return self.numpy(), self.df.index
        return self.df.values

    def pytorch(self):
        data = self.numpy()
        return torch.FloatTensor(data)

    def __len__(self):
        return self.length

    @staticmethod
    def build():
        raise NotImplementedError

    def load_raw(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError
