#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import itertools

def bin_sequence(a, nbins):
    """
    Find and apply the maximum entropy binning to an integer sequence,
    given the number of target bins.

    Arguments:
    ----------
    a: (N, ) ndarray
        input sequence; must be integer

    nbins: int
        number of bins

    Returns:
    --------
    b: (N, ) ndarray
        binned sequence

    """
    assert np.all(np.logical_or(_isinteger(a), np.isnan(a))), "Input has to be integer or integer-like!"

    amin, amax = np.int(np.nanmin(a)), np.int(np.nanmax(a))
    best_h = 0.

    for binning in _generate_bins(range(amin, amax+1), nbins):
        h = _evaluate_binning(a, binning)
        if h > best_h:
            best_h = h
            best_binning = binning

    return _apply_binning(a, best_binning)

def bin_array(A, nbins, axis=None):
    """
    Find and apply the maximum entropy binning to an integer array,
    given the number of target bins.

    Convenience wrapper around bin_sequence().

    Arguments:
    ----------
    A: (N, M) ndarray
        input array; must be integer

    nbins: int
        number of bins

    axis: None or int (default None)
        axis along which to bin;
        if None, the optimal binning is chosen based on all values in the array;

    Returns:
    --------
    B: (N, M) ndarray
        binned array

    """

    if axis is None:
        return bin_sequence(A.ravel(), nbins).reshape(A.shape)
    else:
        return np.apply_along_axis(bin_sequence, axis, A, nbins)

def _generate_bins(seq, nbins):
    """
    Generate all ways to break an arbitrary sequence into n non-empty bins,
    where each bin only contains entries that are contiguous in the original sequence.

    @reference:
    Courtesy of Tim Peters:
    http://stackoverflow.com/questions/39376987/compute-all-ways-to-bin-a-series-of-integers-into-n-bins-where-each-bin-only-co
    """
    base = tuple(seq)
    nbase = len(base)
    for ixs in itertools.combinations(range(1, nbase), nbins - 1):
        yield [base[lo: hi]
               for lo, hi in zip((0,) + ixs, ixs + (nbase,))]

def _apply_binning(a, binning):
    b = np.full_like(a, np.nan, dtype=np.float)
    for ii, some_bin in enumerate(binning):
        for some_number in some_bin:
            b[a==some_number] = ii
    return b.astype(a.dtype)

def _get_h(b):
    counts = np.bincount(b[~np.isnan(b)].astype(np.int))
    pdf = counts.astype(np.float) / np.sum(counts)
    h = -np.sum(np.ma.filled(pdf * np.log2(pdf), fill_value=0))
    return h

def _evaluate_binning(a, binning):
    b = _apply_binning(a, binning)
    h = _get_h(b)
    return h

def _isinteger(x):
    return np.equal(np.mod(x, 1), 0)
