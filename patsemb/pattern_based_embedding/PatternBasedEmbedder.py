

import numpy as np
import numba as nb
import copy
import multiprocessing
from typing import Union, List, Dict, Optional

from patsemb.discretization.Discretizer import Discretizer
from patsemb.discretization.SAXDiscretizer import SAXDiscretizer

from patsemb.pattern_mining.PatternMiner import PatternMiner
from patsemb.pattern_mining.SPMF.QCSP import QCSP
from sklearn.exceptions import NotFittedError


class PatternBasedEmbedder:
    """
    Construct pattern-based embeddings for a (collection of) time series.
    This process consists of two steps:

    1. Mine sequential patterns in symbolic representations of the time
       series. A symbolic representation will be generated for each provided
       window size, and patterns will be mined in symbolic representation
       independently. These results in multi-resolution patterns.

    2. Embed the time series values using the mined sequential patterns,
       which indicates at which positions in the time series a pattern occurs.
       The embedding will consist of one row for each mine pattern and one column
       for each observation in the time series. Therefore, each row corresponds
       to a feature and each column corresponds to a feature vector for a time
       series value.

    Parameters
    ----------
    discretizer: Discretizer, default=SAXDiscretizer()
        The discretizer to convert time series into a symbolic representation
        of discrete symbols.
    pattern_miner: PatternMiner, default=QCSP()
        The pattern miner used to mine sequential patterns in the discrete
        representation
    window_sizes: int or List[int], default=None
        The window sizes to use for discretizing the time series. If ``None`` is
        provided, then the window size of ´´discretizer´´ will be used.
    relative_support_embedding: bool, default=True
        Whether to construct an embedding using the relative support or a
        binary value indicating if the pattern occurs in a subsequence.
    n_jobs: int, default=None
        The number of parallel jobs to use for mining the patterns within the
        time series and constructing the pattern-based embedding.

    Attributes
    ----------
    fitted_discretizers_: Dict[int, Discretizer]
        The fitted discretizers, which can be used for computing a symbolic
        representation of a time series. The key of each item in the dictionary
        equals the window size used for discretization, while the value equals
        the fitted discretizer.
    patterns_: Dict[int, List[np.array]
        The mined sequential patterns. The key of each item in the dictionary
        equals the window size in which the patterns were mined, while the value
        equals the mined patterns.

    References
    ----------
    .. L. Carpentier, L. Feremans, W. Meert, and M. Verbeke.
       "Pattern-based time series semantic segmentation with gradual state transitions".
       In Proceedings of the 2024 SIAM International Conference on Data Mining (SDM),
       pages 316–324. SIAM, april 2024. doi: 10.1137/1.9781611978032.36.
    """

    def __init__(self,
                 discretizer: Discretizer = None,
                 pattern_miner: PatternMiner = None,
                 *,
                 window_sizes: Union[List[int], int] = None,
                 relative_support_embedding: bool = True,
                 n_jobs: Optional[int] = None):
        self.discretizer: Discretizer = discretizer or SAXDiscretizer()
        self.pattern_miner: PatternMiner = pattern_miner or QCSP()
        self.window_sizes: List[int] = \
            [self.discretizer.window_size] if window_sizes is None \
            else [window_sizes] if isinstance(window_sizes, int) \
            else window_sizes
        self.relative_support_embedding: bool = relative_support_embedding
        self.n_jobs: Optional[int] = n_jobs
        self.fitted_discretizers_: List[Dict[int, Discretizer]]
        self.patterns_: List[Dict[int, List[np.array]]]

        self.fitted_discretizers_: Optional[List[Dict[int, Discretizer]]] = None
        self.patterns_: Optional[List[Dict[int, List[np.array]]]] = None

    def fit(self, dataset: Union[np.ndarray, List[np.ndarray]], y=None) -> 'PatternBasedEmbedder':
        """
        Fit this pattern-based embedder using a (collection of) time series.
        This is achieved by mining patterns in the discrete representation of
        the given time series. If multivariate time series are given, then each
        time series must have the same dimension!

        Parameters
        ----------
        dataset: np.ndarray of shape (n_samples, n_attributes) or list of np.ndarray of shape (n_samples, n_attributes)
            The (collection of) time series to use for fitting this pattern-based embedder.
            If a collection of time series is given, then each collection may have a
            different length. For univariate time series, the given numpy arrays may
            be one-dimensional.
        y: Ignored
            Is passed for fitting the discretizer, but will typically not be used and
            is only present here for API consistency by convention.

        Returns
        -------
        self: PatternBasedEmbedder
            Returns the instance itself
        """
        if not isinstance(dataset, List):
            dataset = [dataset]

        # Check the input dimensions
        dimensions = list(get_nb_attributes(time_series) for time_series in dataset)
        if len(set(dimensions)) > 1:
            raise Exception("If a collection of time series is given to PatternBasedEmbedder.fit(), then "
                            "all time series should have the same number of attributes. Now time series with "
                            f"the following number of attributes were provided: {dimensions}")
        dimension = dimensions[0]

        # Initialize the fitted discretizers and patterns
        self.fitted_discretizers_ = [{} for _ in range(dimension)]
        self.patterns_ = [{} for _ in range(dimension)]

        # Create the embedding
        if self.n_jobs is None or self.n_jobs == 1:

            # Treat each attribute & resolution independently
            for attribute in range(dimension):
                attribute_data = [get_attribute(time_series, attribute) for time_series in dataset]
                for window_size in self.window_sizes:
                    # Fit the discretizer
                    discretizer = copy.deepcopy(self.discretizer)
                    discretizer.window_size = window_size
                    discretizer.fit(attribute_data, y)

                    # Convert the dataset to symbolic subsequences
                    discrete_subsequences = np.concatenate([discretizer.transform(time_series) for time_series in attribute_data])

                    # Mine the patterns
                    patterns = self.pattern_miner.mine(discrete_subsequences)

                    # Save the results
                    self.fitted_discretizers_[attribute][window_size] = discretizer
                    self.patterns_[attribute][window_size] = patterns

        else:
            attribute_data = [
                [get_attribute(time_series, attribute) for time_series in dataset]
                for attribute in range(dimension)
            ]
            jobs = [
                (attribute_data[attribute], y, attribute, window_size)
                for attribute in range(dimension)
                for window_size in self.window_sizes
             ]
            with multiprocessing.Pool(processes=min(self.n_jobs, len(jobs))) as pool:
                pool_results = pool.starmap(self._fit_parallel, jobs)

            for attribute, discretizer_dict, patterns_dict in pool_results:
                self.fitted_discretizers_[attribute].update(discretizer_dict)
                self.patterns_[attribute].update(patterns_dict)

        return self

    def _fit_parallel(self, attribute_data: List[np.array], y, attribute: int, window_size: int):
        """
        Executes one sub-process for fitting this PatternBasedEmbedder, i.e., mine the
        patterns in the data of one attribute in one resolution. This method should not
        be used independently.
        """
        # Set and fit the discretizer
        discretizer = copy.deepcopy(self.discretizer)
        discretizer.window_size = window_size
        discretizer.fit(attribute_data, y)

        # Convert the dataset to symbolic subsequences
        discrete_subsequences = np.concatenate([discretizer.transform(time_series) for time_series in attribute_data])

        # Mine the patterns
        patterns = self.pattern_miner.mine(discrete_subsequences)

        # Return all information
        return attribute, {window_size: discretizer}, {window_size: patterns}

    def transform(self, time_series: np.ndarray) -> np.ndarray:
        """
        Transform the given time series into a pattern-based embedding.

        Parameters
        ----------
        time_series: np.array of shape (n_samples, n_attributes)
            The time series to transform into a pattern-based embedding. A
            univariate time series may be one-dimensional.

        Returns
        -------
        pattern_based_embedding: np.ndarray of shape (n_patterns, n_samples)
            The pattern-based embedding, which has a column for each observation in
            the time series and a row for each mined pattern. Each column serves as
            a feature vector for the corresponding time stamp.
        """
        # Check if this pattern-based embedder has been initialized
        if self.patterns_ is None:
            raise NotFittedError("The PatternBasedEmbedder should be fitted before calling the '.transform()' method!")

        # Check if the dimension of the time series matches
        if len(self.patterns_) != get_nb_attributes(time_series):
            raise Exception(f'The given time series has a dimension of {get_nb_attributes(time_series)}, but this embedder '
                            f'has been fitted for {len(self.patterns_)} attributes')

        # Create the pattern-based embedding
        if self.n_jobs is None or self.n_jobs == 1:
            return np.concatenate([
                pattern_based_embedding(
                    self.patterns_[attribute][window_size],
                    self.fitted_discretizers_[attribute][window_size].transform(get_attribute(time_series, attribute)),
                    self.relative_support_embedding,
                    window_size,
                    self.discretizer.stride,
                    time_series.shape[0]
                )
                for attribute in range(get_nb_attributes(time_series))
                for window_size in self.window_sizes
            ])

        else:
            jobs = [
                (
                    self.patterns_[attribute][window_size],
                    self.fitted_discretizers_[attribute][window_size].transform(get_attribute(time_series, attribute)),
                    self.relative_support_embedding,
                    window_size,
                    self.discretizer.stride,
                    time_series.shape[0]
                )
                for attribute in range(get_nb_attributes(time_series))
                for window_size in self.window_sizes
             ]
            with multiprocessing.Pool(processes=min(self.n_jobs, len(jobs))) as pool:
                return np.concatenate(pool.starmap(pattern_based_embedding, jobs))

    def fit_transform(self, time_series: np.ndarray, y=None) -> np.ndarray:
        """
        Fit this PatternBasedEmbedder using the given time series (i.e., mine the
        patterns in the discrete representation of the time series) and immediately
        transform the time series into a pattern-based embedding.

        Parameters
        ----------
        time_series: np.ndarray of shape (n_samples, n_attributes)
            The multivariate time series to transform into a pattern-based embedding.
        y: Ignored
            Is passed for fitting the discretizer, but will typically not be used and
            is only present here for API consistency by convention.

        Returns
        -------
        pattern_based_embedding: np.ndarray of shape (n_patterns, n_samples)
            The pattern-based embedding, which has a column for each observation in
            the time series and a row for each mined pattern. Each column serves as
            a feature vector for the corresponding time stamp.
        """
        return self.fit(time_series, y).transform(time_series)


def pattern_based_embedding(
        patterns: List[np.array],
        discrete_subsequences: np.ndarray,
        relative_support_embedding: bool,
        window_size: int,
        stride: int,
        time_series_length: int) -> np.ndarray:
    """
    Compute the pattern-based embedding for the given patterns and
    discrete subsequences, using the provided information about the
    time series.

    Parameters
    ----------
    patterns: List[np.array]
        The mined sequential patterns to use for creating the embedding
    discrete_subsequences: List[np.array]
        The discrete subsequences of the time series in which the patterns
        be searched.
    relative_support_embedding: bool
        Whether to use the relative support of a pattern to embed a sequence
        or to use a binary value denoting if the pattern occurs.
    window_size: int
        The size of the windows in the original time series.
    stride: int
        The stride used within the sliding to create windows.
    time_series_length: int
        The Length of the original time series.

    Returns
    -------
    pattern_based_embedding: np.array of shape (n_patterns, n_samples)
        The pattern-based embedding, which has a column for each observation in
        the time series and a row for each given pattern. Each column serves as
        a feature vector for the corresponding time stamp.
    """
    # Identify in which windows the patterns occur
    pattern_occurs = np.empty(shape=(len(patterns), len(discrete_subsequences)))
    for pattern_id, pattern in enumerate(patterns):
        for subsequence_id, subsequence in enumerate(discrete_subsequences):
            pattern_occurs[pattern_id, subsequence_id] = pattern_occurs_in_subsequence(pattern, subsequence)

    # Include the relative support if required
    if relative_support_embedding:
        pattern_occurs *= pattern_occurs.mean(axis=1)[:, np.newaxis]

    # Convert to embedding matrix per time step
    embedding_matrix = windowed_to_observation_embedding(pattern_occurs, window_size, stride, time_series_length)

    return embedding_matrix


@nb.njit(fastmath=True)
def pattern_occurs_in_subsequence(pattern: np.array, subsequence: np.array) -> bool:
    """
    Checks whether the given pattern occurs in the given subsequence.

    Parameters
    ----------
    pattern: np.array
        The symbols of the pattern to check.
    subsequence: np.array
        The symbols of the subsequence to check.

    Returns
    -------
    pattern_occurs: bool
        True if and only if the given pattern occurs as an ordered sequence
        in the given subsequence without any gaps.
    """
    length_pattern, length_subsequence = len(pattern), len(subsequence)
    if length_pattern > length_subsequence:  # Quick check
        return False
    for window in np.lib.stride_tricks.sliding_window_view(subsequence, length_pattern):
        if np.array_equal(pattern, window):
            return True
    return False


@nb.njit(fastmath=True)
def windowed_to_observation_embedding(window_based_embedding: np.ndarray, window_size: int, stride: int, time_series_length: int) -> np.ndarray:
    """
    Format the given window-based embedding such that each observation in the
    original time series has exactly one column. If an observation is covered
    by multiple windows, then the average of the embedding of these overlapping
    windows is taken.

    Parameters
    ----------
    window_based_embedding: np.array of shape (n_patterns, n_windows)
        The embedding of each window
    window_size: int
        The size of the windows in the original time series.
    stride: int
        The stride used within the sliding to create windows.
    time_series_length: int
        The Length of the original time series.

    Returns
    -------
    observation_based_embedding: np.array of shape (n_patterns, time_series_length)
        An observation based embedding, such that for each time point in the original
        time series there is exactly one embedding column.
    """
    # Retrieve the boundaries of the windows
    starts_window = np.arange(window_based_embedding.shape[1]) * stride
    ends_window = starts_window + window_size
    ends_window[-1] = time_series_length

    # Iterate over all the time indices, and compute a running sum of the covering windows
    current_start_window, current_end_window = 0, 0
    running_sum = np.zeros(shape=window_based_embedding.shape[0])
    observation_based_embedding = np.empty((window_based_embedding.shape[0], time_series_length))
    for t in range(time_series_length):

        # Add next window to the running sum, if it has been reached
        if current_start_window < starts_window.shape[0] and t == starts_window[current_start_window]:
            running_sum += window_based_embedding[:, current_start_window]
            current_start_window += 1

        # Remove the previous window from the running sum, if it has passed
        if current_end_window < ends_window.shape[0] and t == ends_window[current_end_window]:
            running_sum -= window_based_embedding[:, current_end_window]
            current_end_window += 1

        # Set the embedding for time t as the running sum, divided by the total number of covering windows
        observation_based_embedding[:, t] = running_sum / (current_start_window - current_end_window)

    # Return the observation-based embedding
    return observation_based_embedding


def get_nb_attributes(time_series: np.ndarray) -> int:
    return 1 if len(time_series.shape) == 1 else time_series.shape[1]


def get_attribute(time_series: np.ndarray, attribute: int) -> np.array:
    if not (0 <= attribute < get_nb_attributes(time_series)):
        raise Exception(f'Trying to access attribute {attribute} in {get_nb_attributes(time_series)}-dimensional time series!')
    return time_series if len(time_series.shape) == 1 else time_series[:, attribute]


