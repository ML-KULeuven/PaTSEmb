
import abc
import numpy as np
from typing import List, Union


class Discretizer(abc.ABC):
    """
    Convert a time series to a series of symbolic subsequences, in which
    each subsequence consists of a number of discrete symbols. This is
    achieved by iterating over a time series using a sliding window
    (with given window size and stride), and subsequently convert each
    subsequence to a discrete representation.

    Parameters
    ----------
    window_size: int
        The length of the sliding window, i.e., the sizes of the windows
        to convert to symbolic subsequences.
    stride: int
        The stride for the sliding window, i.e., the amount of time staps
        by which the sliding window shifts to extract the next window.

    See Also
    --------
    SAXDiscretizer: Discretize the time series using SAX
    """

    def __init__(self, window_size: int, stride: int):
        self.window_size: int = window_size
        self.stride: int = stride

    @abc.abstractmethod
    def fit(self, dataset: Union[np.array, List[np.array]], y=None) -> 'Discretizer':
        """
        Fit this discretizer for the given (collection of) time series.

        Parameters
        ----------
        dataset: np.array of shape (n_samples,) or list of np.array of shape (n_samples,)
            The (collection of) time series to use for fitting this discretizer.
        y: Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self: Discretizer
            Returns the instance itself
        """

    @abc.abstractmethod
    def transform(self, time_series: np.array) -> np.ndarray:
        """
        Discretizer the given time series.

        Parameters
        ----------
        time_series: np.array of shape (n_samples,)
            The time series to discretize.

        Returns
        -------
        symbolic_subsequences: np.ndarray of size (n_symbolic_sequences, length_symbolic_sequences)
            The symbolic subsequences as a numpy array, with each row
            representing a different symbolic subsequence.
        """
