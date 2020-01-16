#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import itertools

def bin_sequence(a, nbins):
    """Given a fixed number of bins, find and apply the maximum entropy
    binning to an integer sequence, subject to the constraints that
    - each bin contains a consecutive series of integers,
    - each bin is non-empty,
    - no integer value appears in two bins.

    This function uses the brute-force solution, i.e. it first
    enumerates all possible binnings and then selects the one with the
    highest entropy when applied to the data.

    However, as the approach is exhaustive, it is also often slow.
    For a fast but approximate approach, see bin_sequence_approximately.

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
    Convenience wrapper around the function bin_sequence.

    Given a fixed number of bins, find and apply the maximum entropy
    binning to an integer array, subject to the constraints that
    - each bin contains a consecutive series of integers,
    - each bin is non-empty,
    - no integer value appears in two bins.

    This function uses the brute-force solution, i.e. it first
    enumerates all possible binnings and then selects the one with the
    highest entropy when applied to the data.

    However, as the approach is exhaustive, it is also often slow.
    For a fast but approximate approach, see bin_array_approximately.

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

def bin_sequence_approximately(a, nbins):
    """Given a fixed number of bins, find and apply the maximum entropy
    binning to an integer sequence, subject to the constraints that
    - each bin contains a consecutive series of integers,
    - each bin is non-empty,
    - no integer value appears in two bins.

    This function uses a heuristic to find such a binning and is hence
    only approximate. However, it is also likely to be much faster
    than the exact solution.

    Arguments:
    ----------
    a: (N, ) ndarray
        input sequence; must be integer

    nbins: int
        number of bins; must be a power of two (2, 4, 8, ..., 1024)

    Returns:
    --------
    b: (N, ) ndarray
        binned sequence

    """

    assert np.all(np.logical_or(_isinteger(a), np.isnan(a))), \
        "Input has to be integer or integer-like!"
    assert _isinteger(np.log2(nbins)), \
        "The argument 'nbins' must be a power of two! Current value: {}".format(nbins)

    # split the data into two such that each data split contains half
    # the number of samples; as there may be many duplicates of the
    # median value, we should check with which split we include the
    # median
    pivot = np.median(a)
    is_smaller = a < pivot
    is_larger = a > pivot
    if np.sum(is_smaller) < np.sum(is_larger):
        is_smaller += a == pivot
    else:
        is_larger += a == pivot

    # recurse on each half
    if (np.sum(is_smaller) > 1) and (nbins / 2 > 1):
        smaller = bin_sequence_approximately(a[is_smaller], nbins/2)
    else: # base case
        smaller = np.zeros(np.sum(is_smaller), dtype=np.int)

    if (np.sum(is_larger) > 1) and (nbins / 2 > 1):
        larger = bin_sequence_approximately(a[is_larger], nbins/2)
    else: # base case
        larger = np.zeros(np.sum(is_larger), dtype=np.int) # sic!

    # join results
    b = np.zeros_like(a)
    b[is_smaller] = smaller
    b[is_larger] = larger + np.max(smaller) + 1

    return b

def bin_array_approximately(A, nbins, axis=None):
    """Convenience wrapper around bin_sequence_approximately().

    Given a fixed number of bins, find and apply the maximum entropy
    binning to an integer array, subject to the constraints that
    - each bin contains a consecutive series of integers,
    - each bin is non-empty,
    - no integer value appears in two bins.

    This function uses a heuristic to find such a binning and is hence
    only approximate. However, it is also likely to be much faster
    than the exact solution.

    Arguments:
    ----------
    A: (N, M) ndarray
        input array; must be integer

    nbins: int
        number of bins; must be a power of two (2, 4, 8, ..., 1024)

    axis: None or int (default None)
        axis along which to bin;
        if None, the optimal binning is chosen based on all values in the array;

    Returns:
    --------
    B: (N, M) ndarray
        binned array

    """

    if axis is None:
        return bin_sequence_approximately(A.ravel(), nbins).reshape(A.shape)
    else:
        return np.apply_along_axis(bin_sequence_approximately, axis, A, nbins)
