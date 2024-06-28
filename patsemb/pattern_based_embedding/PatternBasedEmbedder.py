import copy

import numpy as np
import numba as nb
from typing import Union, List, Dict, Optional

from patsemb.discretization.Discretizer import Discretizer
from patsemb.pattern_mining.PatternMiner import PatternMiner
from sklearn.exceptions import NotFittedError


class PatternBasedEmbedder:

    def __init__(self,
                 discretizer: Discretizer,
                 pattern_miner: PatternMiner,
                 window_sizes: List[int] = None,
                 relative_support_embedding: bool = True):
        self.discretizer: Discretizer = discretizer
        self.pattern_miner: PatternMiner = pattern_miner
        self.window_sizes: List[int] = window_sizes or [self.discretizer.window_size]
        self.relative_support_embedding: bool = relative_support_embedding

        self.fitted_discretizers_: Optional[Dict[int, Discretizer]] = None
        self.patterns_: Optional[Dict[int, List[np.array]]] = None

    def fit(self, dataset: Union[np.ndarray, List[np.ndarray]], y=None) -> 'PatternBasedEmbedder':
        """ Mines the frequent patterns in one (or a collection of) time series. """
        self.fitted_discretizers_ = {}
        self.patterns_ = {}
        for window_size in self.window_sizes:
            self.fitted_discretizers_[window_size] = copy.deepcopy(self.discretizer)
            self.fitted_discretizers_[window_size].window_size = window_size
            discrete_subsequences = self.fitted_discretizers_[window_size].fit_transform(dataset, y)
            self.patterns_[window_size] = self.pattern_miner.mine(discrete_subsequences)
        return self

    def transform(self, time_series: np.ndarray) -> np.ndarray:
        """ Creates a pattern-based embedding of one time series """
        if self.patterns_ is None:
            raise NotFittedError("The PatternBasedEmbedder should be fitted before calling the '.transform()' method!")

        embeddings = [
            _pattern_based_embedding(
                self.patterns_[window_size],
                self.fitted_discretizers_[window_size].transform(time_series),
                self.relative_support_embedding,
                self.discretizer.window_size,
                self.discretizer.stride,
                time_series.shape[0]
            )
            for window_size in self.window_sizes
        ]

        return np.concatenate(embeddings)


def _pattern_based_embedding(
        patterns: List[np.array],
        discrete_subsequences: np.ndarray,
        relative_support_embedding: bool,
        window_size: int,
        stride: int,
        time_series_length: int) -> np.ndarray:

    # Identify in which windows the patterns occur
    pattern_occurs = np.empty(shape=(len(patterns), len(discrete_subsequences)))
    for pattern_id, pattern in enumerate(patterns):
        for subsequence_id, subsequence in enumerate(discrete_subsequences):
            pattern_occurs[pattern_id, subsequence_id] = _pattern_occurs_in_subsequence(pattern, subsequence)

    # Include the relative support if required
    if relative_support_embedding:
        pattern_occurs *= pattern_occurs.mean(axis=1)[:, np.newaxis]

    # Convert to embedding matrix per time step
    embedding_matrix = _windowed_to_observation_embedding(pattern_occurs, window_size, stride, time_series_length)

    return embedding_matrix


@nb.njit(fastmath=True)
def _pattern_occurs_in_subsequence(pattern: np.array, subsequence: np.array) -> bool:
    length_pattern, length_subsequence = len(pattern), len(subsequence)
    if length_pattern > length_subsequence:  # Quick check
        return False
    for window in np.lib.stride_tricks.sliding_window_view(subsequence, length_pattern):
        if np.array_equal(pattern, window):
            return True
    return False


@nb.njit(fastmath=True)
def _windowed_to_observation_embedding(window_based_embedding: np.ndarray, window_size: int, stride: int, time_series_length: int) -> np.ndarray:
    """
    Format the embedding such that each individual time unit is embedded in case there are
    overlapping windows due to a stride smaller than the interval length
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


def main():
    cycles = 5  # how many sine cycles
    resolution = 500  # how many datapoints to generate
    length = np.pi * 2 * cycles
    time_series = np.sin(np.arange(0, length, length / resolution))

    import matplotlib.pyplot as plt
    plt.plot(time_series)
    plt.show()

    from patsemb.discretization import SAXDiscretizer
    discretizer = SAXDiscretizer(
        alphabet_size=5,
        word_size=8,
        window_size=16,
        stride=3,
        discretize_within='time_series'
    )

    from patsemb.pattern_mining import QCSP, NOSEP
    pattern_miner = QCSP()

    embedder = PatternBasedEmbedder(discretizer, pattern_miner, window_sizes=[8, 16])
    embedding_matrix = embedder.fit(time_series).transform(time_series)
    plt.imshow(embedding_matrix)
    plt.show()

    # import timeit
    # number = 50
    # total_time = timeit.timeit(lambda: embedder.fit(time_series).transform(time_series), number=number)
    # print(f"Time [s]: {total_time} (over {number} runs, average time: {total_time/number})")


if __name__ == '__main__':
    main()
