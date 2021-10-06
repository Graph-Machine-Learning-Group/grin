import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import haversine_distances


def sample_mask(shape, p=0.002, p_noise=0., max_seq=1, min_seq=1, rng=None):
    if rng is None:
        rand = np.random.random
        randint = np.random.randint
    else:
        rand = rng.random
        randint = rng.integers
    mask = rand(shape) < p
    for col in range(mask.shape[1]):
        idxs = np.flatnonzero(mask[:, col])
        if not len(idxs):
            continue
        fault_len = min_seq
        if max_seq > min_seq:
            fault_len = fault_len + int(randint(max_seq - min_seq))
        idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
        idxs = np.unique(idxs_ext)
        idxs = np.clip(idxs, 0, shape[0] - 1)
        mask[idxs, col] = True
    mask = mask | (rand(mask.shape) < p_noise)
    return mask.astype('uint8')


def compute_mean(x, index=None):
    """Compute the mean values for each datetime. The mean is first computed hourly over the week of the year.
    Further NaN values are computed using hourly mean over the same month through the years. If other NaN are present,
    they are removed using the mean of the sole hours. Hoping reasonably that there is at least a non-NaN entry of the
    same hour of the NaN datetime in all the dataset."""
    if isinstance(x, np.ndarray) and index is not None:
        shape = x.shape
        x = x.reshape((shape[0], -1))
        df_mean = pd.DataFrame(x, index=index)
    else:
        df_mean = x.copy()
    cond0 = [df_mean.index.year, df_mean.index.isocalendar().week, df_mean.index.hour]
    cond1 = [df_mean.index.year, df_mean.index.month, df_mean.index.hour]
    conditions = [cond0, cond1, cond1[1:], cond1[2:]]
    while df_mean.isna().values.sum() and len(conditions):
        nan_mean = df_mean.groupby(conditions[0]).transform(np.nanmean)
        df_mean = df_mean.fillna(nan_mean)
        conditions = conditions[1:]
    if df_mean.isna().values.sum():
        df_mean = df_mean.fillna(method='ffill')
        df_mean = df_mean.fillna(method='bfill')
    if isinstance(x, np.ndarray):
        df_mean = df_mean.values.reshape(shape)
    return df_mean


def geographical_distance(x=None, to_rad=True):
    """
    Compute the as-the-crow-flies distance between every pair of samples in `x`. The first dimension of each point is
    assumed to be the latitude, the second is the longitude. The inputs is assumed to be in degrees. If it is not the
    case, `to_rad` must be set to False. The dimension of the data must be 2.

    Parameters
    ----------
    x : pd.DataFrame or np.ndarray
        array_like structure of shape (n_samples_2, 2).
    to_rad : bool
        whether to convert inputs to radians (provided that they are in degrees).

    Returns
    -------
    distances :
        The distance between the points in kilometers.
    """
    _AVG_EARTH_RADIUS_KM = 6371.0088

    # Extract values of X if it is a DataFrame, else assume it is 2-dim array of lat-lon pairs
    latlon_pairs = x.values if isinstance(x, pd.DataFrame) else x

    # If the input values are in degrees, convert them in radians
    if to_rad:
        latlon_pairs = np.vectorize(np.radians)(latlon_pairs)

    distances = haversine_distances(latlon_pairs) * _AVG_EARTH_RADIUS_KM

    # Cast response
    if isinstance(x, pd.DataFrame):
        res = pd.DataFrame(distances, x.index, x.index)
    else:
        res = distances

    return res


def infer_mask(df, infer_from='next'):
    """Infer evaluation mask from DataFrame. In the evaluation mask a value is 1 if it is present in the DataFrame and
    absent in the `infer_from` month.

    @param pd.DataFrame df: the DataFrame.
    @param str infer_from: denotes from which month the evaluation value must be inferred.
    Can be either `previous` or `next`.
    @return: pd.DataFrame eval_mask: the evaluation mask for the DataFrame
    """
    mask = (~df.isna()).astype('uint8')
    eval_mask = pd.DataFrame(index=mask.index, columns=mask.columns, data=0).astype('uint8')
    if infer_from == 'previous':
        offset = -1
    elif infer_from == 'next':
        offset = 1
    else:
        raise ValueError('infer_from can only be one of %s' % ['previous', 'next'])
    months = sorted(set(zip(mask.index.year, mask.index.month)))
    length = len(months)
    for i in range(length):
        j = (i + offset) % length
        year_i, month_i = months[i]
        year_j, month_j = months[j]
        mask_j = mask[(mask.index.year == year_j) & (mask.index.month == month_j)]
        mask_i = mask_j.shift(1, pd.DateOffset(months=12 * (year_i - year_j) + (month_i - month_j)))
        mask_i = mask_i[~mask_i.index.duplicated(keep='first')]
        mask_i = mask_i[np.in1d(mask_i.index, mask.index)]
        eval_mask.loc[mask_i.index] = ~mask_i.loc[mask_i.index] & mask.loc[mask_i.index]
    return eval_mask


def prediction_dataframe(y, index, columns=None, aggregate_by='mean'):
    """Aggregate batched predictions in a single DataFrame.

    @param (list or np.ndarray) y: the list of predictions.
    @param (list or np.ndarray) index: the list of time indexes coupled with the predictions.
    @param (list or pd.Index) columns: the columns of the returned DataFrame.
    @param (str or list) aggregate_by: how to aggregate the predictions in case there are more than one for a step.
    - `mean`: take the mean of the predictions
    - `central`: take the prediction at the central position, assuming that the predictions are ordered chronologically
    - `smooth_central`: average the predictions weighted by a gaussian signal with std=1
    - `last`: take the last prediction
    @return: pd.DataFrame df: the evaluation mask for the DataFrame
    """
    dfs = [pd.DataFrame(data=data.reshape(data.shape[:2]), index=idx, columns=columns) for data, idx in zip(y, index)]
    df = pd.concat(dfs)
    preds_by_step = df.groupby(df.index)
    # aggregate according passed methods
    aggr_methods = ensure_list(aggregate_by)
    dfs = []
    for aggr_by in aggr_methods:
        if aggr_by == 'mean':
            dfs.append(preds_by_step.mean())
        elif aggr_by == 'central':
            dfs.append(preds_by_step.aggregate(lambda x: x[int(len(x) // 2)]))
        elif aggr_by == 'smooth_central':
            from scipy.signal import gaussian
            dfs.append(preds_by_step.aggregate(lambda x: np.average(x, weights=gaussian(len(x), 1))))
        elif aggr_by == 'last':
            dfs.append(preds_by_step.aggregate(lambda x: x[0]))  # first imputation has missing value in last position
        else:
            raise ValueError('aggregate_by can only be one of %s' % ['mean', 'central' 'smooth_central', 'last'])
    if isinstance(aggregate_by, str):
        return dfs[0]
    return dfs


def ensure_list(obj):
    if isinstance(obj, (list, tuple)):
        return list(obj)
    else:
        return [obj]


def missing_val_lens(mask):
    m = np.concatenate([np.zeros((1, mask.shape[1])),
                        (~mask.astype('bool')).astype('int'),
                        np.zeros((1, mask.shape[1]))])
    mdiff = np.diff(m, axis=0)
    lens = []
    for c in range(m.shape[1]):
        mj, = mdiff[:, c].nonzero()
        diff = np.diff(mj)[::2]
        lens.extend(list(diff))
    return lens


def disjoint_months(dataset, months=None, synch_mode='window'):
    idxs = np.arange(len(dataset))
    months = ensure_list(months)
    # divide indices according to window or horizon
    if synch_mode == 'window':
        start, end = 0, dataset.window - 1
    elif synch_mode == 'horizon':
        start, end = dataset.horizon_offset, dataset.horizon_offset + dataset.horizon - 1
    else:
        raise ValueError('synch_mode can only be one of %s' % ['window', 'horizon'])
    # after idxs
    start_in_months = np.in1d(dataset.index[dataset._indices + start].month, months)
    end_in_months = np.in1d(dataset.index[dataset._indices + end].month, months)
    idxs_in_months = start_in_months & end_in_months
    after_idxs = idxs[idxs_in_months]
    # previous idxs
    months = np.setdiff1d(np.arange(1, 13), months)
    start_in_months = np.in1d(dataset.index[dataset._indices + start].month, months)
    end_in_months = np.in1d(dataset.index[dataset._indices + end].month, months)
    idxs_in_months = start_in_months & end_in_months
    prev_idxs = idxs[idxs_in_months]
    return prev_idxs, after_idxs


def thresholded_gaussian_kernel(x, theta=None, threshold=None, threshold_on_input=False):
    if theta is None:
        theta = np.std(x)
    weights = np.exp(-np.square(x / theta))
    if threshold is not None:
        mask = x > threshold if threshold_on_input else weights < threshold
        weights[mask] = 0.
    return weights
