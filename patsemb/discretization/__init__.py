
"""
This module provides the functionality to convert a time series into a symbolic representation.
and can be imported as follows:

>>> from patsemb import discretization

The symbolic representation of a time series is a set of symbolic words, constructed using a
fixed size alphabet. Because the symbolic words do not consist of continuous values (which is
the case for time series), they are suitable for mining sequential patterns.
"""


from .Discretizer import Discretizer
from .SAXDiscretizer import SAXDiscretizer

__all__ = [
    'Discretizer',
    'SAXDiscretizer'
]
