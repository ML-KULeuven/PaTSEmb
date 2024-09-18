
"""
The pattern-based embedding modules offers to functionality to construct a
pattern-based embedding. It only requires a few lines of code to embed a
time series via the :py:class:`~patsemb.pattern_based_embedding.PatternBasedEmbedder`.
First, you need to import the package as follows

>>> from patsemb import pattern_based_embedding

Next, let us initialize a random time series using numpy.

>>> import numpy as np
>>> time_series = np.random.rand(1000)

It now only takes two lines to embed the time series: one line to initialize
the pattern-based embedder, and one line to call the fit_transform method!

>>> pattern_based_embedder = pattern_based_embedding.PatternBasedEmbedder()
>>> embedding = pattern_based_embedder.fit_transform(time_series)
"""

from .PatternBasedEmbedder import PatternBasedEmbedder
from .visualization import plot_time_series_and_embedding, plot_time_series, plot_embedding

__all__ = [
    'PatternBasedEmbedder',
    'plot_time_series_and_embedding',
    'plot_time_series',
    'plot_embedding'
]
