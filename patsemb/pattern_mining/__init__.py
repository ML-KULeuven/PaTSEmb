
"""
This package provides the functionality to mining sequential patterns in
a symbolic representation of the time series. The pattern mining algorithms
can be imported as follows:

>>> from patsemb import pattern_mining

All mining algorithms inherit from the :py:class:`~patsemb.pattern_mining.PatternMiner`
class, which can be used to mine sequential patterns via the :py:func:`~patsemb.pattern_mining.PatternMiner.mine`
function.
"""

from .PatternMiner import PatternMiner
from .SPMF import SPMF, NOSEP, QCSP

__all__ = [
    'PatternMiner',
    'SPMF',
    'NOSEP',
    'QCSP'
]
