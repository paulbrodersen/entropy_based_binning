# Entropy-based binning for discrete variables

In data analysis and machine learning, it is often necessary to bin
the values of a variable as a preprocessing step. The binning that
retains the largest amount of information about the original ranks of
the data is the binning that results in the (discrete) uniform
distribution, as the uniform distribution is the maximum entropy
distribution for a variable on a finite domain.

For continuous variables, finding a mapping that results in evenly
filled bins is trivial: simply partition the data points according to
their ranks (terciles to result in three bins, quartiles to obtain 4
bins, etc).

For categorical variables, the problem of finding the grouping of
categories that results in the most evenly filled bins reduces to the
(multi-way) [partition
problem](https://en.wikipedia.org/wiki/Partition_problem), which is
known to be NP-hard. However, many approximate solutions exist
(for examples, see [Korf
(2009)](https://www.ijcai.org/Proceedings/09/Papers/096.pdf)).

For discrete, ordinal data neither set of approaches is very
desirable. Treating discrete, ordinal data as categorical may result
in mappings in which values, which are far apart in the original
space, are grouped together in one bin. Conversely, ranking can result
in many ties, and if one enforces a discrete uniform target
distribution, data points with the same rank can end up in different
bins, which is generally also undesirable.

Unfortunately, there appears to be no simple algorithm to place bin
edges between uniquely valued data points that guarantees a
distribution that is as close as possible to the desired discrete
uniform distribution.

Some people employ a recursive approach, in which on each iteration
they partition the data into the two most similarly sized partitions
and then recurse on each partition. However, this approach only works
for a number of bins that is a power of 2, and furthermore is not
guaranteed to converge to the optimal solution (for my original use
case, the solutions were terrible, which sparked the creation of this
module).

This module implements the functionality to exhaustively search for
the highest entropy binning of a sequence of integers, such that
1. each bin maps back to a sequence of consecutive integers,
2. consecutive integers are either in the same bin or in consecutive bins, and
2. no two bins contain the same integer.

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
