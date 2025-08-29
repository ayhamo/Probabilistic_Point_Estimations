import math
import pickle
import warnings
from types import SimpleNamespace

import numpy as np
import sklearn.model_selection

from tdgplib.helper.traintest import TrainTestSplit


def scale_data(x, y):
    scaler = SimpleNamespace(x=sklearn.preprocessing.StandardScaler(), y=sklearn.preprocessing.StandardScaler())
    scaler.x.fit(x.train)
    x = x.apply(scaler.x.transform)
    scaler.y.fit(y.train)
    y = y.apply(scaler.y.transform)
    return x, y, scaler


def get_folds(x_all, y_all, n_fold, random_seed):
    kfold = sklearn.model_selection.KFold(max(2, n_fold), shuffle=True, random_state=random_seed)
    # noinspection PyArgumentList
    folds = [
        [TrainTestSplit(x_all[train], x_all[test]), TrainTestSplit(y_all[train], y_all[test])]
        for train, test in kfold.split(x_all, y_all)
    ]
    return folds[:n_fold]


def get_parabolas(random_seed, n_fold=2):
    s = 0.4
    n = 500
    rng = np.random.default_rng(19960111)
    m1, m2 = np.array([[-1, 1], [2, 1]])
    x1 = rng.multivariate_normal(m1, s * np.eye(2), size=n // 2)
    x2 = rng.multivariate_normal(m2, s * np.eye(2), size=n // 2)
    y1 = x1[:, 0] ** 2 + x1[:, 0]
    y2 = x2[:, 1] ** 2 + x2[:, 1]

    x = np.concatenate([x1, x2], axis=0)
    y = np.concatenate([y1, y2], axis=0)[:, None]
    for x, y in get_folds(x, y, n_fold=n_fold, random_seed=random_seed):
        x, y, scalers = scale_data(x, y)
        yield x, y, scalers


def get_sinc(random_seed, n_fold=2, num_data=350):
    rng = np.random.default_rng(19960111)

    x = rng.random((num_data, 2)) * 2 - 1
    w = np.stack([
        np.sin(3.1414 * x[..., 0]) * x[..., 0] * 2,
        np.cos(3.1414 * x[..., 0]) * 2,
    ], axis=-1)[..., None]
    z = (x[:, None] @ w)[..., 0, 0]
    y = (np.sinc(z) - z ** 2)[:, None]
    for x, y in get_folds(x, y, n_fold=n_fold, random_seed=random_seed):
        x, y, scalers = scale_data(x, y)
        yield x, y, scalers


def from_pickle(path, random_seed, n_fold=2):
    """
    Returns the dataset in `path` relative to the current working directory
    """
    with open(path, "rb") as f:
        x, y = pickle.load(f)
    n_fold = max(2, n_fold)
    if x.shape[0] > (1500*n_fold)/(n_fold - 1):
        sub_size = math.floor((1500*n_fold)/(n_fold - 1))
        rng = np.random.default_rng(random_seed)
        warnings.warn(f'Dataset of size {x.shape[0]} is too large, subsampling to {sub_size}')
        idx = rng.choice(len(x), sub_size, replace=False)
        x, y = x[idx], y[idx]

    y = y.reshape(-1, 1)
    for x, y in get_folds(x, y, n_fold, random_seed):
        x, y, scalers = scale_data(x, y)
        yield x, y, scalers
