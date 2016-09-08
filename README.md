# Entropy-based binning:

Exhaustively search for the highest entropy binning of a sequence of integers,
and apply the binning. Use to bin variables that are

(1) integer (or at least discrete), and  
(2) have a 'natural' order (ordinal).  

Typical examples include such things like age, tax bands, etc.  

Do not use for binning floats / continuous variables, as there is a
much easier way to find a good binning (i.e. map data points to
terciles/quartiles/quintiles/percentiles/etc. of the data).

# Example:

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
