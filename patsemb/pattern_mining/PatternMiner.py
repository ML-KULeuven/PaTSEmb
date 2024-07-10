import abc
import numpy as np
from typing import List


class PatternMiner(abc.ABC):
    """
    Mine patterns in a discrete representation of the time series.

    See Also
    --------
    SPMF: Mine frequent sequential patterns
    """

    @abc.abstractmethod
    def mine(self, discrete_sequences: np.ndarray) -> List[np.array]:
        """
        Fit this discretizer for the given (collection of) time series.

        Parameters
        ----------
        discrete_sequences: np.array of shape (n_symbolic_sequences, length_symbolic_sequences)
            The discrete representation of a time series. This representation
            consists of ´n_symbolic_sequences´ subsequences, each one having
            ´length_symbolic_sequences´ symbols. The sequences are provided
            as the rows of the given input matrix.

        Returns
        -------
        self: PatternMiner
            Returns the instance itself
        """
