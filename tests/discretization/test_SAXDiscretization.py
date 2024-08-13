
import pytest
import string
import random
import numpy as np
import scipy
from sklearn.exceptions import NotFittedError
from patsemb.discretization.SAXDiscretizer import SAXDiscretizer, compute_bins, segment_time_series, discretize


def generate_random_univariate_time_series() -> np.array:
    return np.random.rand(np.random.choice([500, 1000, 5000]))


class TestSAXDiscretizer:

    @pytest.mark.parametrize('alphabet_size', [3, 5, 7])
    @pytest.mark.parametrize('word_size', [8, 12, 16])
    @pytest.mark.parametrize('window_size', [16, 32, 64])
    @pytest.mark.parametrize('stride', [1, 4, 8])
    @pytest.mark.parametrize('discretize_within', ['window', 'time_series', 'complete'])
    def test_initialization(self, alphabet_size, word_size, window_size, stride, discretize_within):
        discretizer = SAXDiscretizer(
            alphabet_size=alphabet_size,
            word_size=word_size,
            window_size=window_size,
            stride=stride,
            discretize_within=discretize_within
        )
        assert discretizer.alphabet_size == alphabet_size
        assert discretizer.word_size == word_size
        assert discretizer.window_size == window_size
        assert discretizer.stride == stride
        assert discretizer.discretize_within == discretize_within

    @pytest.mark.parametrize('seed', range(20))
    def test_invalid_initialization(self, seed):
        random.seed(seed)
        random_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=random.randint(1, 20)))
        if random_string not in ['window', 'time_series', 'complete']:
            with pytest.raises(Exception):
                SAXDiscretizer(discretize_within=random_string)

    def test_compute_bins_simple(self):
        assert np.array_equal(compute_bins(np.array([1, 2, 3, 4, 5]), alphabet_size=2), np.array([-np.inf, 3, np.inf]))
        assert np.array_equal(compute_bins(np.array([1, 2, 4, 5]), alphabet_size=2), np.array([-np.inf, 3, np.inf]))
        assert np.array_equal(compute_bins(np.array([1, 2, 3, 5]), alphabet_size=2), np.array([-np.inf, 2.75, np.inf]))
        mean = np.mean([1, 2, 3, 4, 5])
        std = np.std([1, 2, 3, 4, 5])
        assert np.array_equal(compute_bins(np.array([1, 2, 3, 4, 5]), alphabet_size=4),
                              np.array([-np.inf, scipy.stats.norm(mean, std).ppf(0.25), 3, scipy.stats.norm(mean, std).ppf(0.75), np.inf]))

    @pytest.mark.parametrize('alphabet_size', [2, 3, 4, 5, 6, 7])
    @pytest.mark.parametrize('seed', range(10))
    def test_compute_bins(self, alphabet_size, seed):
        np.random.seed(seed)
        time_series = generate_random_univariate_time_series()
        bins = compute_bins(time_series, alphabet_size)
        assert bins[0] == -np.inf
        assert bins[-1] == np.inf
        assert np.all(bins[:-1] < bins[1:])  # Check if bins are sorted

        standard_normal_time_series = (time_series - time_series.mean()) / time_series.std()
        bins = compute_bins(standard_normal_time_series, alphabet_size)

        relative = 0.1
        if alphabet_size == 2:
            assert bins[1] == pytest.approx(0.0, relative)

        elif alphabet_size == 3:
            assert bins[1] == pytest.approx(-0.43, relative)
            assert bins[2] == pytest.approx(0.43, relative)

        elif alphabet_size == 4:
            assert bins[1] == pytest.approx(-0.67, relative)
            assert bins[2] == pytest.approx(0.0, relative)
            assert bins[3] == pytest.approx(0.67, relative)

        elif alphabet_size == 5:
            assert bins[1] == pytest.approx(-0.84, relative)
            assert bins[2] == pytest.approx(-0.25, relative)
            assert bins[3] == pytest.approx(0.25, relative)
            assert bins[4] == pytest.approx(0.84, relative)

        elif alphabet_size == 6:
            assert bins[1] == pytest.approx(-0.97, relative)
            assert bins[2] == pytest.approx(-0.43, relative)
            assert bins[3] == pytest.approx(0.0, relative)
            assert bins[4] == pytest.approx(0.43, relative)
            assert bins[5] == pytest.approx(0.97, relative)

        elif alphabet_size == 7:
            assert bins[1] == pytest.approx(-1.07, relative)
            assert bins[2] == pytest.approx(-0.57, relative)
            assert bins[3] == pytest.approx(-0.18, relative)
            assert bins[4] == pytest.approx(0.18, relative)
            assert bins[5] == pytest.approx(0.57, relative)
            assert bins[6] == pytest.approx(1.07, relative)

    @pytest.mark.parametrize('alphabet_size', [2, 3, 4, 5, 6, 7])
    @pytest.mark.parametrize('seed', range(10))
    def test_compute_bins_invariant_to_order(self, alphabet_size, seed):
        np.random.seed(seed)
        time_series = generate_random_univariate_time_series()
        bins = compute_bins(time_series, alphabet_size)
        np.random.shuffle(time_series)
        bins_shuffled = compute_bins(time_series, alphabet_size)
        assert np.allclose(bins, bins_shuffled)

    @pytest.mark.parametrize('window_size', [8, 16, 32])
    @pytest.mark.parametrize('stride', [1, 4, 8, 16])
    @pytest.mark.parametrize('word_size', [4, 7, 8])  # Also an 'annoying', not nicely dividable value
    @pytest.mark.parametrize('seed', range(3))
    def test_segment_time_series(self, window_size, stride, word_size, seed):
        np.random.seed(seed)
        time_series = generate_random_univariate_time_series()
        segments = segment_time_series(time_series, window_size, stride, word_size)

        assert segments.shape[0] == (time_series.shape[0] - window_size) // stride + 1
        assert segments.shape[1] == word_size

        for i, segment in enumerate(segments):
            window = time_series[i*stride: i*stride + window_size] if i < segments.shape[0] - 1 else time_series[i*stride:]
            base_split_size = window.shape[0] // word_size
            offset = 0
            for j in range(window.shape[0] % word_size):
                assert segment[j] == pytest.approx(np.mean(window[offset:offset + base_split_size + 1]))
                offset += base_split_size + 1
            for j in range(window.shape[0] % word_size, word_size):
                assert segment[j] == pytest.approx(np.mean(window[offset:offset + base_split_size]))
                offset += base_split_size

    @pytest.mark.parametrize('alphabet_size', [2, 3, 4, 5])
    @pytest.mark.parametrize('seed', range(3))
    def test_discretize(self, alphabet_size, seed):
        np.random.seed(seed)
        time_series = generate_random_univariate_time_series()
        bins = np.sort(np.insert(np.random.uniform(size=alphabet_size - 1), [0, alphabet_size - 1], [-np.inf, np.inf]))
        discrete = discretize(time_series, bins)

        bin_map = [None for _ in range(alphabet_size)]
        for value, symbol in zip(time_series, discrete):
            for bin_id in range(alphabet_size):
                if not bins[bin_id] <= value < bins[bin_id + 1]:
                    continue
                if bin_map[bin_id] is None:
                    bin_map[bin_id] = symbol
                else:
                    assert bin_map[bin_id] == symbol

        nb_empty_bins = 0
        for symbol in bin_map:
            if symbol is None:
                nb_empty_bins += 1

        assert discrete.shape == time_series.shape
        assert np.unique(discrete).shape[0] == alphabet_size - nb_empty_bins

    @pytest.mark.parametrize('seed', range(3))
    def test_transform_invalid_within(self, seed):
        discretizer = SAXDiscretizer(discretize_within='time_series')  # Because within time series is fastest
        np.random.seed(seed)
        time_series = generate_random_univariate_time_series()
        try:
            discretizer.transform(time_series)
        except AttributeError:
            pytest.fail('An error is raised while this should not happen')
        while discretizer.discretize_within in ['window', 'time_series', 'complete']:
            discretizer.discretize_within = ''.join(random.choices(string.ascii_uppercase + string.digits, k=random.randint(1, 20)))
        with pytest.raises(AttributeError):
            discretizer.transform(time_series)


class TestSAXDiscretizerWindow:

    @pytest.fixture
    def discretizer(self) -> SAXDiscretizer:
        return SAXDiscretizer(discretize_within='window')

    @pytest.mark.parametrize('seed', range(15))
    def test_fit(self, discretizer, seed):
        np.random.seed(seed)
        time_series = generate_random_univariate_time_series()
        discretizer.fit(time_series)
        assert discretizer.bins_ is None

    @pytest.mark.parametrize('seed', range(3))
    def test_transform_not_fitted(self, discretizer, seed):
        np.random.seed(seed)
        time_series = generate_random_univariate_time_series()
        try:
            discretizer.transform(time_series)
        except NotFittedError:
            pytest.fail('An error is raised while this should not happen')

    def test_transform(self, discretizer):
        discretizer.stride = 16
        time_series = np.arange(80)
        discrete = discretizer.transform(time_series)
        assert discrete.shape == (5, 8)
        assert np.unique(discrete).shape[0] == 5
        assert np.array_equal(discrete[0, :], [1, 1, 2, 3, 3, 4, 5, 5])
        assert np.array_equal(discrete[1, :], [1, 1, 2, 3, 3, 4, 5, 5])
        assert np.array_equal(discrete[2, :], [1, 1, 2, 3, 3, 4, 5, 5])
        assert np.array_equal(discrete[3, :], [1, 1, 2, 3, 3, 4, 5, 5])
        assert np.array_equal(discrete[4, :], [1, 1, 2, 3, 3, 4, 5, 5])
        assert discrete.dtype == int


class TestSAXDiscretizerTimeSeries:

    @pytest.fixture
    def discretizer(self) -> SAXDiscretizer:
        return SAXDiscretizer(discretize_within='time_series')

    @pytest.mark.parametrize('seed', range(15))
    def test_fit(self, discretizer, seed):
        np.random.seed(seed)
        time_series = generate_random_univariate_time_series()
        discretizer.fit(time_series)
        assert discretizer.bins_ is None

    @pytest.mark.parametrize('seed', range(3))
    def test_transform_not_fitted(self, discretizer, seed):
        np.random.seed(seed)
        time_series = generate_random_univariate_time_series()
        try:
            discretizer.transform(time_series)
        except NotFittedError:
            pytest.fail('An error is raised while this should not happen')

    def test_transform(self, discretizer):
        discretizer.stride = 16
        time_series = np.arange(80)
        discrete = discretizer.transform(time_series)
        assert discrete.shape == (5, 8)
        assert np.unique(discrete).shape[0] == 5
        assert np.array_equal(discrete[0, :], [1, 1, 1, 1, 1, 1, 1, 1])
        assert np.array_equal(discrete[1, :], [1, 1, 2, 2, 2, 2, 2, 2])
        assert np.array_equal(discrete[2, :], [2, 3, 3, 3, 3, 3, 3, 4])
        assert np.array_equal(discrete[3, :], [4, 4, 4, 4, 4, 4, 5, 5])
        assert np.array_equal(discrete[4, :], [5, 5, 5, 5, 5, 5, 5, 5])
        assert discrete.dtype == int


class TestSAXDiscretizerComplete:

    @pytest.fixture
    def discretizer(self) -> SAXDiscretizer:
        return SAXDiscretizer(discretize_within='complete')

    @pytest.mark.parametrize('alphabet_size', [2, 3, 4, 5])
    @pytest.mark.parametrize('seed', range(10))
    def test_fit(self, discretizer, alphabet_size, seed):
        discretizer.alphabet_size = alphabet_size
        np.random.seed(seed)
        if np.random.rand() < 0.2:
            time_series = generate_random_univariate_time_series()
            all_values = np.array([value for value in time_series])

        else:
            time_series = [
                generate_random_univariate_time_series()
                for _ in range(np.random.randint(2, 10))
            ]
            all_values = np.array([value for attribute in time_series for value in attribute])

        discretizer.fit(time_series)
        assert discretizer.bins_ is not None

        bins = compute_bins(all_values, alphabet_size)
        assert np.array_equal(bins, discretizer.bins_)

    @pytest.mark.parametrize('seed', range(3))
    def test_transform_not_fitted(self, discretizer, seed):
        np.random.seed(seed)
        time_series = generate_random_univariate_time_series()
        with pytest.raises(NotFittedError):
            discretizer.transform(time_series)

    def test_transform(self, discretizer):
        discretizer.stride = 16
        time_series = [np.arange(80), np.arange(80, 160)]
        discretizer.fit(time_series)

        discrete = discretizer.transform(time_series[0])
        assert discrete.shape == (5, 8)
        assert np.unique(discrete).shape[0] == 3
        assert np.array_equal(discrete[0, :], [1, 1, 1, 1, 1, 1, 1, 1])
        assert np.array_equal(discrete[1, :], [1, 1, 1, 1, 1, 1, 1, 1])
        assert np.array_equal(discrete[2, :], [1, 1, 1, 1, 1, 2, 2, 2])
        assert np.array_equal(discrete[3, :], [2, 2, 2, 2, 2, 2, 2, 2])
        assert np.array_equal(discrete[4, :], [2, 2, 3, 3, 3, 3, 3, 3])
        assert discrete.dtype == int

        discrete = discretizer.transform(time_series[1])
        assert discrete.shape == (5, 8)
        assert np.unique(discrete).shape[0] == 3
        assert np.array_equal(discrete[0, :], [3, 3, 3, 3, 3, 3, 4, 4])
        assert np.array_equal(discrete[1, :], [4, 4, 4, 4, 4, 4, 4, 4])
        assert np.array_equal(discrete[2, :], [4, 4, 4, 5, 5, 5, 5, 5])
        assert np.array_equal(discrete[3, :], [5, 5, 5, 5, 5, 5, 5, 5])
        assert np.array_equal(discrete[4, :], [5, 5, 5, 5, 5, 5, 5, 5])
        assert discrete.dtype == int

        discretizer.fit(time_series[0])
        discrete = discretizer.transform(time_series[1])
        assert discrete.shape == (5, 8)
        assert np.unique(discrete).shape[0] == 1
        assert np.array_equal(discrete[0, :], [5, 5, 5, 5, 5, 5, 5, 5])
        assert np.array_equal(discrete[1, :], [5, 5, 5, 5, 5, 5, 5, 5])
        assert np.array_equal(discrete[2, :], [5, 5, 5, 5, 5, 5, 5, 5])
        assert np.array_equal(discrete[3, :], [5, 5, 5, 5, 5, 5, 5, 5])
        assert np.array_equal(discrete[4, :], [5, 5, 5, 5, 5, 5, 5, 5])
        assert discrete.dtype == int

