from abc import ABC, abstractmethod
import numpy as np


class AbstractScaler(ABC):

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        params = ", ".join([f"{k}={str(v)}" for k, v in self.params().items()])
        return "{}({})".format(self.__class__.__name__, params)

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def params(self):
        return {k: v for k, v in self.__dict__.items() if not callable(v) and not k.startswith("__")}

    @abstractmethod
    def fit(self, x):
        pass

    @abstractmethod
    def transform(self, x):
        pass

    @abstractmethod
    def inverse_transform(self, x):
        pass

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def to_torch(self):
        import torch
        for p in self.params():
            param = getattr(self, p)
            param = np.atleast_1d(param)
            param = torch.tensor(param).float()
            setattr(self, p, param)
        return self


class Scaler(AbstractScaler):
    def __init__(self, offset=0., scale=1.):
        self.bias = offset
        self.scale = scale
        super(Scaler, self).__init__()

    def params(self):
        return dict(bias=self.bias, scale=self.scale)

    def fit(self, x, mask=None, keepdims=True):
        pass

    def transform(self, x):
        return (x - self.bias) / self.scale

    def inverse_transform(self, x):
        return x * self.scale + self.bias

    def fit_transform(self, x, mask=None, keepdims=True):
        self.fit(x, mask, keepdims)
        return self.transform(x)


class StandardScaler(Scaler):
    def __init__(self, axis=0):
        self.axis = axis
        super(StandardScaler, self).__init__()

    def fit(self, x, mask=None, keepdims=True):
        if mask is not None:
            x = np.where(mask, x, np.nan)
            self.bias = np.nanmean(x, axis=self.axis, keepdims=keepdims)
            self.scale = np.nanstd(x, axis=self.axis, keepdims=keepdims)
        else:
            self.bias = x.mean(axis=self.axis, keepdims=keepdims)
            self.scale = x.std(axis=self.axis, keepdims=keepdims)
        return self


class MinMaxScaler(Scaler):
    def __init__(self, axis=0):
        self.axis = axis
        super(MinMaxScaler, self).__init__()

    def fit(self, x, mask=None, keepdims=True):
        if mask is not None:
            x = np.where(mask, x, np.nan)
            self.bias = np.nanmin(x, axis=self.axis, keepdims=keepdims)
            self.scale = (np.nanmax(x, axis=self.axis, keepdims=keepdims) - self.bias)
        else:
            self.bias = x.min(axis=self.axis, keepdims=keepdims)
            self.scale = (x.max(axis=self.axis, keepdims=keepdims) - self.bias)
        return self
