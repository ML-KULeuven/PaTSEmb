
"""
Frequent, sequential patterns are sequential patterns that often
occur in the set of symbolic words. This ensures that the mined
patterns are correspond to typical shapes of the time series. Extensive
research in frequent pattern mining resulted in numerous pattern-mining
algorithms. We rely on the SPMF-library [FLGG16]_ to mine frequent
sequential patterns.

.. [FLGG16] Fournier-Viger, P., Lin, C.W., Gomariz, A., Gueniche, T., Soltani,
   A., Deng, Z., Lam, H. T. (2016). The SPMF Open-Source Data Mining
   Library Version 2. Proc. 19th European Conference on Principles of
   Data Mining and Knowledge Discovery (PKDD 2016) Part III.
   https://doi.org/10.1007/978-3-319-46131-1_8
"""

from .SPMF import SPMF
from .NOSEP import NOSEP
from .QCSP import QCSP

__all__ = [
    'SPMF',
    'NOSEP',
    'QCSP'
]
