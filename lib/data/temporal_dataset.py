import numpy as np
import pandas as pd
import torch
from einops import rearrange
from pandas import DatetimeIndex
from torch.utils.data import Dataset

from .preprocessing import AbstractScaler


class TemporalDataset(Dataset):
    def __init__(self, data,
                 index=None,
                 freq=None,
                 exogenous=None,
                 trend=None,
                 scaler=None,
                 window=24,
                 horizon=24,
                 delay=0,
                 stride=1):
        """Wrapper class for dataset whose entry are dependent from a sequence of temporal indices.

        Parameters
        ----------
        data : np.ndarray
            Data relative to the main signal.
        index : DatetimeIndex or None
            Temporal indices for the data.
        exogenous : dict or None
            Exogenous data and label paired with main signal (default is None).
        trend : np.ndarray or None
            Trend paired with main signal (default is None). Must be of the same length of 'data'.
        scaler : AbstractScaler or None
            Scaler that must be used for data (default is None).
        freq : pd.DateTimeIndex.freq or str
            Frequency of the indices (defaults is indices.freq).
        window : int
            Size of the sliding window in the past.
        horizon : int
            Size of the prediction horizon.
        delay : int
            Offset between end of window and start of horizon.

        Raises
        ----------
        ValueError
            If a frequency for the temporal indices is not provided neither in indices nor explicitly.
            If preprocess is True and data_scaler is None.
        """
        super(TemporalDataset, self).__init__()
        # Initialize signatures
        self.__exogenous_keys = dict()
        self.__reserved_signature = {'data', 'trend', 'x', 'y'}
        # Store data
        self.data = data
        if exogenous is not None:
            for name, value in exogenous.items():
                self.add_exogenous(value, name, for_window=True, for_horizon=True)
        # Store time information
        self.index = index
        try:
            freq = freq or index.freq or index.inferred_freq
            self.freq = pd.tseries.frequencies.to_offset(freq)
        except AttributeError:
            self.freq = None
        # Store offset information
        self.window = window
        self.delay = delay
        self.horizon = horizon
        self.stride = stride
        # Identify the indices of the samples
        self._indices = np.arange(self.data.shape[0] - self.sample_span + 1)[::self.stride]
        # Store preprocessing options
        self.trend = trend
        self.scaler = scaler

    def __getitem__(self, item):
        return self.get(item, self.preprocess)

    def __contains__(self, item):
        return item in self.__exogenous_keys

    def __len__(self):
        return len(self._indices)

    def __repr__(self):
        return "{}(n_samples={})".format(self.__class__.__name__, len(self))

    # Getter and setter for data

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, value):
        assert value is not None
        self.__data = self.check_input(value)

    @property
    def trend(self):
        return self.__trend

    @trend.setter
    def trend(self, value):
        self.__trend = self.check_input(value)

    # Setter for exogenous data

    def add_exogenous(self, obj, name, for_window=True, for_horizon=False):
        assert isinstance(name, str)
        if name.endswith('_window'):
            name = name[:-7]
            for_window, for_horizon = True, False
        if name.endswith('_horizon'):
            name = name[:-8]
            for_window, for_horizon = False, True
        if name in self.__reserved_signature:
            raise ValueError("Channel '{0}' cannot be added in this way. Use obj.{0} instead.".format(name))
        if not (for_window or for_horizon):
            raise ValueError("Either for_window or for_horizon must be True.")
        obj = self.check_input(obj)
        setattr(self, name, obj)
        self.__exogenous_keys[name] = dict(for_window=for_window, for_horizon=for_horizon)
        return self

    # Dataset properties

    @property
    def horizon_offset(self):
        return self.window + self.delay

    @property
    def sample_span(self):
        return max(self.horizon_offset + self.horizon, self.window)

    @property
    def preprocess(self):
        return (self.trend is not None) or (self.scaler is not None)

    @property
    def n_steps(self):
        return self.data.shape[0]

    @property
    def n_channels(self):
        return self.data.shape[-1]

    @property
    def indices(self):
        return self._indices

    # Signature information

    @property
    def exo_window_keys(self):
        return {k for k, v in self.__exogenous_keys.items() if v['for_window']}

    @property
    def exo_horizon_keys(self):
        return {k for k, v in self.__exogenous_keys.items() if v['for_horizon']}

    @property
    def exo_common_keys(self):
        return self.exo_window_keys.intersection(self.exo_horizon_keys)

    @property
    def signature(self):
        attrs = []
        if self.window > 0:
            attrs.append('x')
            for attr in self.exo_window_keys:
                attrs.append(attr if attr not in self.exo_common_keys else (attr + '_window'))
        for attr in self.exo_horizon_keys:
            attrs.append(attr if attr not in self.exo_common_keys else (attr + '_horizon'))
        attrs.append('y')
        attrs = tuple(attrs)
        preprocess = []
        if self.trend is not None:
            preprocess.append('trend')
        if self.scaler is not None:
            preprocess.extend(self.scaler.params())
        preprocess = tuple(preprocess)
        return dict(data=attrs, preprocessing=preprocess)

    # Item getters

    def get(self, item, preprocess=False):
        idx = self._indices[item]
        res, transform = dict(), dict()
        if self.window > 0:
            res['x'] = self.data[idx:idx + self.window]
            for attr in self.exo_window_keys:
                key = attr if attr not in self.exo_common_keys else (attr + '_window')
                res[key] = getattr(self, attr)[idx:idx + self.window]
        for attr in self.exo_horizon_keys:
            key = attr if attr not in self.exo_common_keys else (attr + '_horizon')
            res[key] = getattr(self, attr)[idx + self.horizon_offset:idx + self.horizon_offset + self.horizon]
        res['y'] = self.data[idx + self.horizon_offset:idx + self.horizon_offset + self.horizon]
        if preprocess:
            if self.trend is not None:
                y_trend = self.trend[idx + self.horizon_offset:idx + self.horizon_offset + self.horizon]
                res['y'] = res['y'] - y_trend
                transform['trend'] = y_trend
                if 'x' in res:
                    res['x'] = res['x'] - self.trend[idx:idx + self.window]
            if self.scaler is not None:
                transform.update(self.scaler.params())
                if 'x' in res:
                    res['x'] = self.scaler.transform(res['x'])
        return res, transform

    def snapshot(self, indices=None, preprocess=True):
        if not self.preprocess:
            preprocess = False
        data, prep = [{k: [] for k in sign} for sign in self.signature.values()]
        indices = np.arange(len(self._indices)) if indices is None else indices
        for idx in indices:
            data_i, prep_i = self.get(idx, preprocess)
            [v.append(data_i[k]) for k, v in data.items()]
            if len(prep_i):
                [v.append(prep_i[k]) for k, v in prep.items()]
        data = {k: np.stack(ds) for k, ds in data.items() if len(ds)}
        if len(prep):
            prep = {k: np.stack(ds) if k == 'trend' else ds[0] for k, ds in prep.items() if len(ds)}
        return data, prep

    # Data utilities

    def expand_indices(self, indices=None, unique=False, merge=False):
        ds_indices = dict.fromkeys([time for time in ['window', 'horizon'] if getattr(self, time) > 0])
        indices = np.arange(len(self._indices)) if indices is None else indices
        if 'window' in ds_indices:
            w_idxs = [np.arange(idx, idx + self.window) for idx in self._indices[indices]]
            ds_indices['window'] = np.concatenate(w_idxs)
        if 'horizon' in ds_indices:
            h_idxs = [np.arange(idx + self.horizon_offset, idx + self.horizon_offset + self.horizon)
                      for idx in self._indices[indices]]
            ds_indices['horizon'] = np.concatenate(h_idxs)
        if unique:
            ds_indices = {k: np.unique(v) for k, v in ds_indices.items()}
        if merge:
            ds_indices = np.unique(np.concatenate(list(ds_indices.values())))
        return ds_indices

    def overlapping_indices(self, idxs1, idxs2, synch_mode='window', as_mask=False):
        assert synch_mode in ['window', 'horizon']
        ts1 = self.data_timestamps(idxs1, flatten=False)[synch_mode]
        ts2 = self.data_timestamps(idxs2, flatten=False)[synch_mode]
        common_ts = np.intersect1d(np.unique(ts1), np.unique(ts2))
        is_overlapping = lambda sample: np.any(np.in1d(sample, common_ts))
        m1 = np.apply_along_axis(is_overlapping, 1, ts1)
        m2 = np.apply_along_axis(is_overlapping, 1, ts2)
        if as_mask:
            return m1, m2
        return np.sort(idxs1[m1]), np.sort(idxs2[m2])

    def data_timestamps(self, indices=None, flatten=True):
        ds_indices = self.expand_indices(indices, unique=False)
        ds_timestamps = {k: self.index[v] for k, v in ds_indices.items()}
        if not flatten:
            ds_timestamps = {k: np.array(v).reshape(-1, getattr(self, k)) for k, v in ds_timestamps.items()}
        return ds_timestamps

    def reduce_dataset(self, indices, inplace=False):
        if not inplace:
            from copy import deepcopy
            dataset = deepcopy(self)
        else:
            dataset = self
        old_index = dataset.index[dataset._indices[indices]]
        ds_indices = dataset.expand_indices(indices, merge=True)
        dataset.index = dataset.index[ds_indices]
        dataset.data = dataset.data[ds_indices]
        if dataset.mask is not None:
            dataset.mask = dataset.mask[ds_indices]
        if dataset.trend is not None:
            dataset.trend = dataset.trend[ds_indices]
        for attr in dataset.exo_window_keys.union(dataset.exo_horizon_keys):
            if getattr(dataset, attr, None) is not None:
                setattr(dataset, attr, getattr(dataset, attr)[ds_indices])
        dataset._indices = np.flatnonzero(np.in1d(dataset.index, old_index))
        return dataset

    def check_input(self, data):
        if data is None:
            return data
        data = self.check_dim(data)
        data = data.clone().detach() if isinstance(data, torch.Tensor) else torch.tensor(data)
        # cast data
        if torch.is_floating_point(data):
            return data.float()
        elif data.dtype in [torch.int, torch.int8, torch.int16, torch.int32, torch.int64]:
            return data.int()
        return data

    # Class-specific methods (override in children)

    @staticmethod
    def check_dim(data):
        if data.ndim == 1:  # [steps] -> [steps, features]
            data = rearrange(data, '(s f) -> s f', f=1)
        elif data.ndim != 2:
            raise ValueError(f'Invalid data dimensions {data.shape}')
        return data

    def dataframe(self):
        return pd.DataFrame(data=self.data, index=self.index)

    @staticmethod
    def add_argparse_args(parser, **kwargs):
        parser.add_argument('--window', type=int, default=24)
        parser.add_argument('--horizon', type=int, default=24)
        parser.add_argument('--delay', type=int, default=0)
        parser.add_argument('--stride', type=int, default=1)
        return parser
