# coding: utf-8

import numpy as np
import pandas as pd
import os
from scipy import stats


def proportion_range(n):
    """Return a list of n equally-spaced proportions between 0 and 1."""
    return [i/n for i in range(0, n)]


def drop_keys(dictionary, keys):
    keys = set(keys)
    return {k: v for k, v in dictionary.items() if k not in keys}


def select_keys(dictionary, keys):
    keys = set(keys)
    return {k: v for k, v in dictionary.items() if k in keys}


def map_seq(func, seq):
    """Take a Series/np-array/list and return a new copy with func applied
    to each sequence item."""
    if isinstance(seq, pd.Series):
        return seq.apply(func)
    else:
        return [func(val) for val in seq]


def mask_iterable(xs, mask, other):
    """Take a Series/np-array/list and return a new copy with all indexes
    where mask is True replaced with the other value."""
    if isinstance(xs, pd.Series):
        return xs.mask(mask, other)
    else:
        xs_series = pd.Series(xs)
        mask_series = pd.Series(mask).reset_index(drop=True)
        masked = xs_series.mask(mask_series, other)
        return masked.to_numpy() if isinstance(xs, np.ndarray) else list(masked)


def make_dir_if_not_exists(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def corrected_resampled_t_test(xs, ys, test_size, alternative='two-sided'):
    """
    Based on corrected resampled t-test as documented in
    Chapter 5, Credibility: Evaluating What’s Been
    Learned, Data Mining: Practical Machine Learning Tools and
    Techniques, Third Edition, 2011.

    Original proposition of correction: https://doi.org/10.1023/A:1024068626366
    Useful discussion of alternatives: https://machinelearningmastery.com/statistical-significance-tests-for-comparing-machine-learning-algorithms/
    """
    train_size = 1 - test_size
    xs, ys = np.array(xs), np.array(ys)
    assert xs.shape == ys.shape
    ds = xs - ys
    d_mean = np.mean(ds)
    # Using same ddof as: https://github.com/Waikato/weka-3.8/blob/49865490cef763855ede07cd11331a7aeaecd110/weka/src/main/java/weka/experiment/Stats.java#L316
    d_var = np.var(ds, ddof=1)
    k = ds.shape[0]
    t = d_mean / np.sqrt(((1 / k) + (test_size / train_size)) * d_var)
    # 2-sided t-test (so multiply by 2) with k-1 degrees of freedom
    if alternative == 'two-sided':
        p = (1.0 - stats.t.cdf(abs(t), k-1)) * 2.0
    elif alternative == 'greater':
        p = (1.0 - stats.t.cdf(t, k-1))
    elif alternative == 'less':
        p = (1.0 - stats.t.cdf(-t, k-1))
    else:
        raise ValueError('Unsupported alternative value: {}'.format(alternative))
    return p
