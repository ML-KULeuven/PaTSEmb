import abc
import numpy as np
from typing import List


class PatternMiner(abc.ABC):
    """
    Mine patterns in a discrete representation of the time series.
    """

    @abc.abstractmethod
    def mine(self, discrete_sequences: np.ndarray, y=None) -> List[np.array]:
        """
        Fit this discretizer for the given (collection of) time series.

        Parameters
        ----------
        discrete_sequences: np.array of shape (n_symbolic_sequences, length_symbolic_sequences)
            The discrete representation of a time series. This representation
            consists of ´n_symbolic_sequences´ subsequences, each one having
            ´length_symbolic_sequences´ symbols. The sequences are provided
            as the rows of the given input matrix.
        y: Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self: List[np.array]
            The list of mined patterns.
        """
