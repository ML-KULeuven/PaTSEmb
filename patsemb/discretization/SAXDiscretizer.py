
import numpy as np
import numba as nb
import scipy
from typing import List, Union
from sklearn.exceptions import NotFittedError

from patsemb.discretization.Discretizer import Discretizer


class SAXDiscretizer(Discretizer):

    def __init__(self,
                 alphabet_size: int = 5,
                 word_size: int = 8,
                 window_size: int = 16,
                 stride: int = 1,
                 discretize_within: str = 'time_series'):
        super().__init__(window_size, stride)
        self.alphabet_size: int = alphabet_size
        self.word_size: int = word_size
        self.discretize_within: str = discretize_within
        self.bins_: np.array = None

        if self.discretize_within not in ['window', 'time_series', 'complete']:
            raise Exception(
                f"Invalid value for 'within' given: '{discretize_within}'\n"
                f"Only valid values are: ['window', 'time_series', 'complete']"
            )

    def fit(self, dataset: Union[np.array, List[np.array]], y=None) -> 'SAXDiscretizer':
        if self.discretize_within == 'complete':
            if isinstance(dataset, List):
                dataset = np.concatenate(dataset, axis=0)
            self.bins_ = compute_bins(dataset, self.alphabet_size)
        return self

    def transform(self, time_series: np.array) -> np.ndarray:

        # Check if this SAX-discretizer is fitted (if necessary)
        if self.bins_ is None and self.discretize_within == 'complete':
            raise NotFittedError('The SAXDiscretizer is not fitted yet! If you want to discretize using the '
                                 'discretization level of a collection of time series (i.e., if discretize_within='
                                 'complete), then the SAXDiscretizer should be fitted first!')

        segments = segment_time_series(time_series, self.window_size, self.stride, self.word_size)
        discrete_segments = np.empty_like(segments)

        if self.discretize_within == 'window':
            for i, segment in enumerate(segments):
                bins = compute_bins(segment, self.alphabet_size)
                discrete_segments[i, :] = discretize(segment, bins)

        elif self.discretize_within == 'time_series':
            bins = compute_bins(time_series, self.alphabet_size)
            for i, segment in enumerate(segments):
                discrete_segments[i, :] = discretize(segment, bins)

        elif self.discretize_within == 'complete':
            for i, segment in enumerate(segments):
                discrete_segments[i, :] = discretize(segment, self.bins_)

        else:
            raise AttributeError(f"The value for 'discretize_within' is invalid: '{self.discretize_within}'")

        return discrete_segments


def compute_bins(time_series: np.array, alphabet_size: int) -> np.array:
    random_variable = scipy.stats.norm(loc=time_series.mean(), scale=time_series.std())
    ppf_inputs = np.linspace(0, 1, alphabet_size + 1)
    return random_variable.ppf(ppf_inputs)


@nb.njit(fastmath=True)
def segment_time_series(time_series: np.array, window_size: int, stride: int, word_size: int) -> np.ndarray:
    # TODO can this be improved? especially the second part
    # Already applies PAA
    nb_segments = ((time_series.shape[0] - window_size) // stride) + 1
    start_segments = np.arange(nb_segments) * stride
    end_segments = start_segments + window_size
    end_segments[-1] = time_series.shape[0]
    discrete_subsequences = np.empty(shape=(nb_segments, word_size))
    for segment_id in range(nb_segments):
        segment = time_series[start_segments[segment_id]:end_segments[segment_id]]
        split_means = [split.mean() for split in np.array_split(segment, word_size)]
        discrete_subsequences[segment_id, :] = split_means
    return discrete_subsequences


@nb.njit(fastmath=True)
def discretize(segment: np.array, bins: np.array) -> np.array:
    return np.digitize(segment, bins)
