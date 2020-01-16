#!/usr/bin/env python
# -*- coding: utf-8 -*-

# entropy_based_binning.py --- Exhaustively search for the highest entropy binning of a sequence of integers.

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

Exhaustively search for the highest entropy binning of a sequence of integers,
and apply the binning. Use to bin variables that are

(1) integer (or at least discrete), and
(2) have a 'natural' order (ordinal).

Typical examples include such things like age, tax bands, etc.

Do not use for binning floats / continuous variables, as there is a
much easier way to find a good binning (i.e. map data points to
terciles/quartiles/quintiles/percentiles/etc. of the data).

Example:
--------

import numpy as np
import entropy_based_binning as ebb

A = np.random.randint(0, 5, size=(10, 100))
B = ebb.bin_array(A, nbins=3, axis=1)
b = ebb.bin_sequence(A[0], nbins=3)

If the data is discrete but not integer, map the data to integers first:

D = np.random.choice(np.linspace(0., 1., 11), size=(10, 100))
_, A = np.unique(D, return_inverse=True)
A = A.reshape(D.shape)
B = ebb.bin_array(A, nbins=3, axis=1)

"""

__version__ = "0.0.1"
__author__  = "Paul Brodersen"
__email__   = "paulbrodersen+ebb@gmail.com"

from ._main import (
    bin_sequence,
    bin_sequence_approximately,
    bin_array,
    bin_array_approximately,
)
