
import numpy as np
import numba as nb
import scipy
from typing import List, Union, Dict, Callable

from patsemb.discretization.Discretizer import Discretizer


class SAXDiscretizer(Discretizer):

    def __init__(self, alphabet_size: int, word_size: int, window_size: int, stride: int, discretize_within: str):
        super().__init__(window_size, stride)
        self.alphabet_size: int = alphabet_size
        self.word_size: int = word_size
        self.discretize_within: str = discretize_within
        self.bins_: np.array = None

        # TODO if the methods are numbafied, then this will probably have to be different
        self.__discretize_within_strategies: Dict[str, Dict[str, Callable]] = {
            'window': {
                'fit': self._fit_within_window,
                'transform': self._transform_within_window
            },
            'time_series': {
                'fit': self._fit_within_time_series,
                'transform': self._transform_within_time_series
            },
            'complete': {
                'fit': self._fit_within_complete,
                'transform': self._transform_within_complete
            }
        }
        if self.discretize_within not in self.__discretize_within_strategies:
            raise Exception(
                f"Invalid value for 'within' given: '{discretize_within}'\n"
                f"Only valid values are: {self.__discretize_within_strategies.keys()}"
            )

    #####################################################################################
    # PUBLIC API
    #####################################################################################

    def fit(self, dataset: Union[np.ndarray, List[np.ndarray]], y=None) -> 'SAXDiscretizer':
        return self.__discretize_within_strategies[self.discretize_within]['fit'](dataset)

    def transform(self, dataset: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        if isinstance(dataset, List):
            discrete_subsequences = []
            for time_series in dataset:
                new_subsequences = self.__discretize_within_strategies[self.discretize_within]['transform'](time_series)
                discrete_subsequences.extend(new_subsequences)
            return discrete_subsequences
        else:
            return self.__discretize_within_strategies[self.discretize_within]['transform'](dataset)

    #####################################################################################
    # FITTING
    #####################################################################################

    def _fit_within_window(self, dataset: Union[np.ndarray, List[np.ndarray]]) -> 'SAXDiscretizer':
        return self

    def _fit_within_time_series(self, dataset: Union[np.ndarray, List[np.ndarray]]) -> 'SAXDiscretizer':
        return self

    def _fit_within_complete(self, dataset: Union[np.ndarray, List[np.ndarray]]) -> 'SAXDiscretizer':
        self.bins_ = self._compute_bins(dataset)
        return self

    #####################################################################################
    # TRANSFORMING
    #####################################################################################

    # TODO numbafy
    def _compute_bins(self, data: Union[np.ndarray, List[np.ndarray]]) -> np.array:
        if isinstance(data, List):
            data = np.vstack(data)
        random_variable = scipy.stats.norm(loc=data.mean(), scale=data.std())
        ppf_inputs = np.linspace(0, 1, self.alphabet_size + 1)
        return random_variable.ppf(ppf_inputs)

    # TODO numbafy
    def _segment(self, time_series: np.array) -> np.ndarray:
        return segment_time_series(
            time_series=time_series,
            window_size=self.window_size,
            stride=self.stride,
            word_size=self.word_size
        )

    # TODO numbafy
    def _transform_within_window(self, time_series: np.array) -> List[np.array]:
        segments = self._segment(time_series)
        discrete_segments = []
        for segment in segments:
            bins = self._compute_bins(segment)
            discrete_segments.append(np.digitize(segment, bins))
        return discrete_segments

    # TODO numbafy
    def _transform_within_time_series(self, time_series: np.array) -> List[np.array]:
        bins = self._compute_bins(time_series)
        return [np.digitize(segment, bins) for segment in self._segment(time_series)]

    # TODO numbafy
    def _transform_within_complete(self, time_series: np.array) -> List[np.array]:
        return [np.digitize(segment, self.bins_) for segment in self._segment(time_series)]


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


def main():
    discretizer = SAXDiscretizer(
        alphabet_size=5,
        word_size=8,
        window_size=17,
        stride=3,
        discretize_within='time_series'
    )

    ts = np.random.normal(size=10000)

    import timeit
    print(timeit.timeit(lambda: discretizer.fit_transform(ts), number=20))


if __name__ == '__main__':
    main()
