
"""
This module offers functionality to postprocess the pattern-based embedding. All
postprocessors can be imported as follows:

>>> from patsemb import postprocess

The postprocessors use the fit-transform interface to handle the pattern-based
embedding. Given a pattern-based embedding, the postprocessors transform it into
a new, slightly adjusted embedding.
"""

from .Postprocessor import Postprocessor
from .Smoother import Smoother

__all__ = [
    'Postprocessor',
    'Smoother'
]
