#!/usr/bin/env python
# -*- coding: utf-8 -*-

# entropy_based_binning.py ---

# Copyright (C) 2016 Paul Brodersen <paulbrodersen+ebb@gmail.com>

# Author: Paul Brodersen <paulbrodersen+ebb@gmail.com>

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""
Entropy-based binning:
----------------------

Exhaustive search for the highest entropy binning of a sequence of integers.
Use to bin variables that are

(1) integer (discrete, ordinal), and
(2) have a 'natural' order.

Typical examples include such things like age, tax bands, etc.

Do not use for binning floats / continuous variables, as there is a
much easier way to find a good binning (i.e. partition into
terciles/quartiles/quintiles/percentiles, etc.).

Example:
--------

import numpy as np
import entropy_based_binning as ebb

A = np.random.randint(0, 5, size=(10, 100))
B = ebb.bin_array(A, nbins=3, axis=1)

"""

import numpy as np
import itertools

def bin_sequence(a, nbins):
    """
    Find and apply the maximum entropy binning to an integer sequence,
    given the number of target bins.

    Arguments:
    ----------
    a: (N, ) ndarray
        input sequence

    nbins: int
        number of bins

    """
    amin, amax = np.nanmin(a), np.nanmax(a)
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
    a: (N, M) ndarray
        input array

    nbins: int
        number of bins

    axis: None or int (default None)
        axis along which to bin

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
    b = np.zeros_like(a)
    for ii, some_bin in enumerate(binning):
        for some_number in some_bin:
            b[a==some_number] = ii
    return b

def _get_h(b):
    counts = np.bincount(b).astype(np.float)
    pdf = counts / np.sum(counts)
    h = -np.sum(np.ma.filled(pdf * np.log2(pdf), fill_value=0))
    return h

def _evaluate_binning(a, binning):
    b = _apply_binning(a, binning)
    h = _get_h(b)
    return h
