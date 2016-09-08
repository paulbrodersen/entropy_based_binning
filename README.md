# Entropy-based binning for discrete variables

In data analysis and machine learning, it is often necessary to bin
the values of a variable as a preprocessing step. The binning that
retains the largest amount of information about the original ranks of
the data is the discrete uniform distribution, which is the maximum
entropy distribution for a discrete variable on a finite domain.

For continuous variables, finding a mapping that results in evenly
filled bins is trivial: simply partition the data points according to
their ranks (terciles to result in three bins, quartiles to obtain 4
bins, etc).

For discrete data, ranking can result in many ties, and if one were to
enforce a discrete uniform target distribution, data points with the
same rank would end up in different bins, which is generally
undesirable. Unfortunately, there appears to be no simple rule to place
bin edges between uniquely valued data points that guarantees a
distribution that is as close as possible to the desired discrete
uniform distribution.

This module implements the functionality to exhaustively search for
the highest entropy binning of a sequence of integers, and to apply
the binning.

## Examples

```python
import numpy as np
import entropy_based_binning as ebb

A = np.random.randint(0, 5, size=(10, 100))
B = ebb.bin_array(A, nbins=3, axis=1)
b = ebb.bin_sequence(A[0], nbins=3)
```

If the data is discrete but not integer, map the data to integers first:

```python
D = np.random.choice(np.linspace(0., 1., 11), size=(10, 100))
_, A = np.unique(D, return_inverse=True)
A = A.reshape(D.shape)
B = ebb.bin_array(A, nbins=3, axis=1)
```
