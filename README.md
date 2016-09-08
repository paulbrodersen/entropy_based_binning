# Entropy-based binning:

Exhaustive search for the highest entropy binning of a sequence of integers.
Use to bin variables that are

(1) integer (discrete, ordinal), and
(2) have a 'natural' order.

Typical examples include such things like age, tax bands, etc.

Do not use for binning floats / continuous variables, as there is a
much easier way to find a good binning (i.e. partition into
terciles/quartiles/quintiles/percentiles, etc.).

# Example:

import numpy as np
import entropy_based_binning as ebb

A = np.random.randint(0, 5, size=(10, 100))  
B = ebb.bin_array(A, nbins=3, axis=1)  
b = ebb.bin_sequence(A[0], nbins=3)  
