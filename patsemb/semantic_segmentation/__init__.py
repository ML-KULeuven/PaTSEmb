
"""
This module offers functionality to compute a semantic segmentation from
a pattern-based embedding. It can be imported as follows:

>>> from patsemb import semantic_segmentation

Currently, only a probabilistic semantic segmentor is implemented. This segmentor
uses the fit-predict_proba interface, because it predicts segment probabilities
instead of segment labels.
"""

from .ProbabilisticSemanticSegmentor import ProbabilisticSemanticSegmentor
from .LogisticRegressionSegmentor import LogisticRegressionSegmentor

__all__ = [
    'ProbabilisticSemanticSegmentor',
    'LogisticRegressionSegmentor'
]
