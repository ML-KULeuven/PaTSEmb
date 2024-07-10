
import pytest
import numpy as np
from sklearn.exceptions import NotFittedError

from patsemb.discretization.SAXDiscretizer import SAXDiscretizer
from patsemb.pattern_mining.SPMF.QCSP import QCSP
from patsemb.pattern_mining.SPMF.NOSEP import NOSEP
from patsemb.pattern_based_embedding.PatternBasedEmbedder import PatternBasedEmbedder, pattern_based_embedding, pattern_occurs_in_subsequence, windowed_to_observation_embedding


def generate_random_univariate_time_series() -> np.array:
    return np.random.rand(np.random.choice([500, 1000, 5000]))


class TestPatternBasedEmbedder:

    def test_default_initialization(self):
        pattern_based_embedder = PatternBasedEmbedder()
        assert isinstance(pattern_based_embedder.discretizer, SAXDiscretizer)
        assert isinstance(pattern_based_embedder.pattern_miner, QCSP)
        assert pattern_based_embedder.window_sizes == [pattern_based_embedder.discretizer.window_size]
        assert pattern_based_embedder.relative_support_embedding

    @pytest.mark.parametrize('discretize_within', ['time_series', 'window', 'complete'])
    @pytest.mark.parametrize('pattern_mining_algorithm', ['QCSP', 'NOSEP'])
    @pytest.mark.parametrize('window_sizes', [16, 32, 64, [16, 32, 64], None])
    @pytest.mark.parametrize('relative_support_embedding', [True, False])
    def test_initialization(self, discretize_within, pattern_mining_algorithm, window_sizes, relative_support_embedding):
        discretizer = SAXDiscretizer(discretize_within=discretize_within)
        pattern_miner = NOSEP() if pattern_mining_algorithm == 'NOSEP' else QCSP()
        pattern_based_embedder = PatternBasedEmbedder(
            discretizer=discretizer,
            pattern_miner=pattern_miner,
            window_sizes=window_sizes,
            relative_support_embedding=relative_support_embedding
        )
        assert pattern_based_embedder.discretizer == discretizer
        assert isinstance(pattern_based_embedder.discretizer, SAXDiscretizer)
        assert pattern_based_embedder.discretizer.discretize_within == discretize_within
        assert pattern_based_embedder.pattern_miner == pattern_miner
        assert isinstance(pattern_based_embedder.pattern_miner, NOSEP if pattern_mining_algorithm == 'NOSEP' else QCSP)
        if window_sizes is None:
            assert pattern_based_embedder.window_sizes == [pattern_based_embedder.discretizer.window_size]
        else:
            assert pattern_based_embedder.window_sizes == (window_sizes if isinstance(window_sizes, list) else [window_sizes])
        assert pattern_based_embedder.relative_support_embedding == relative_support_embedding

        assert pattern_based_embedder.fitted_discretizers_ is None
        assert pattern_based_embedder.patterns_ is None

    @pytest.mark.parametrize('nb_time_series', [1, 3])
    @pytest.mark.parametrize('stride', [1, 5])
    @pytest.mark.parametrize('window_sizes', [16, 64, [16, 32, 64], None])
    @pytest.mark.parametrize('seed', range(3))
    def test_fit_transform(self, nb_time_series, stride, window_sizes, seed):
        np.random.seed(seed)
        embedder = PatternBasedEmbedder(window_sizes=window_sizes)
        embedder.discretizer.stride = stride

        assert embedder.patterns_ is None
        assert embedder.fitted_discretizers_ is None

        if nb_time_series == 1:
            embedder.fit(generate_random_univariate_time_series())
        else:
            embedder.fit([generate_random_univariate_time_series() for _ in range(nb_time_series)])

        assert embedder.patterns_ is not None
        assert embedder.fitted_discretizers_ is not None

        if window_sizes is None:
            assert len(embedder.patterns_) == 1
            assert len(embedder.fitted_discretizers_) == 1
            assert embedder.discretizer.window_size in embedder.patterns_
            assert isinstance(embedder.patterns_[embedder.discretizer.window_size], list)
            assert embedder.discretizer.window_size in embedder.fitted_discretizers_
            assert isinstance(embedder.fitted_discretizers_[embedder.discretizer.window_size], type(embedder.discretizer))

        elif isinstance(window_sizes, int):
            assert len(embedder.patterns_) == 1
            assert len(embedder.fitted_discretizers_) == 1
            assert window_sizes in embedder.patterns_
            assert isinstance(embedder.patterns_[window_sizes], list)
            assert window_sizes in embedder.fitted_discretizers_
            assert isinstance(embedder.fitted_discretizers_[window_sizes], type(embedder.discretizer))

        else:
            assert len(embedder.patterns_) == len(window_sizes)
            assert len(embedder.fitted_discretizers_) == len(window_sizes)
            for window_size in window_sizes:
                assert window_size in embedder.patterns_
                assert isinstance(embedder.patterns_[window_size], list)
                assert window_size in embedder.fitted_discretizers_
                assert isinstance(embedder.fitted_discretizers_[window_size], type(embedder.discretizer))

        test_time_series = generate_random_univariate_time_series()
        embedding = embedder.transform(test_time_series)
        nb_patterns = sum([len(pattern_list) for pattern_list in embedder.patterns_.values()])
        assert embedding.shape[0] == nb_patterns
        assert embedding.shape[1] == test_time_series.shape[0]

    @pytest.mark.parametrize('seed', range(15))
    def test_transform_not_fitted(self, seed):
        np.random.seed(seed)
        time_series = generate_random_univariate_time_series()
        embedder = PatternBasedEmbedder()
        with pytest.raises(NotFittedError):
            embedder.transform(time_series)

    def test_pattern_based_embedding_simple(self):
        patterns = [np.array([1, 2, 3])]
        discrete_subsequences = np.array([
            [1, 2, 3, 3],
            [1, 1, 2, 3],
            [1, 2, 1, 3],
            [1, 2, 3, 1],
        ])

        embedding = pattern_based_embedding(patterns, discrete_subsequences, True, 5, 1, 8)
        expected_embedding = np.array([
            [0.75, 2*0.75/2, 2*0.75/3, 3*0.75/4, 3*0.75/4, 2*0.75/3, 1*0.75/2, 0.75]
        ])
        assert embedding.shape == expected_embedding.shape
        assert np.array_equal(embedding, expected_embedding)

    @pytest.mark.parametrize('nb_symbols', [3, 5])
    @pytest.mark.parametrize('nb_patterns', [10, 25])
    @pytest.mark.parametrize('relative_support_embedding', [True, False])
    @pytest.mark.parametrize('window_size', [16, 64])
    @pytest.mark.parametrize('stride', [1, 8])
    @pytest.mark.parametrize('time_series_length', [500, 1000])
    @pytest.mark.parametrize('seed', range(5))
    def test_pattern_based_embedding(self, nb_symbols, nb_patterns, relative_support_embedding, window_size, stride, time_series_length, seed):
        np.random.seed(seed)
        patterns = [np.random.choice(nb_symbols, np.random.randint(4, 8)) for _ in range(nb_patterns)]
        nb_subsequences = (time_series_length - window_size) // stride + 1
        discrete_subsequences = np.random.choice(nb_symbols, (nb_subsequences, np.random.randint(8, 24)))

        embedding = pattern_based_embedding(patterns, discrete_subsequences, relative_support_embedding, window_size, stride, time_series_length)
        expected_windowed_embedding = np.array([
            [pattern_occurs_in_subsequence(pattern, subsequence) for subsequence in discrete_subsequences]
            for pattern in patterns
        ]).astype(float)
        if relative_support_embedding:
            for i in range(nb_patterns):
                relative_support = expected_windowed_embedding[i, :].sum() / nb_subsequences
                assert 0 <= relative_support <= 1
                for j in range(nb_subsequences):
                    if expected_windowed_embedding[i, j] == 1:
                        expected_windowed_embedding[i, j] = relative_support

        observation_based_embedding = windowed_to_observation_embedding(expected_windowed_embedding, window_size, stride, time_series_length)
        assert embedding.shape == observation_based_embedding.shape
        assert np.array_equal(embedding, observation_based_embedding)

    @pytest.mark.parametrize('nb_symbols', [3, 4, 5, 6])
    @pytest.mark.parametrize('length_pattern', [4, 8, 12])
    @pytest.mark.parametrize('seed', range(25))
    def test_pattern_occurs_in_subsequence_pattern_occurs(self, nb_symbols, length_pattern, seed):
        np.random.seed(seed)
        subsequence = np.random.choice(nb_symbols, max(length_pattern, np.random.choice([10, 15, 20, 25])))
        start_pos = np.random.choice(subsequence.shape[0] - length_pattern + 1)
        pattern = subsequence[start_pos:start_pos+length_pattern]
        assert pattern.shape[0] == length_pattern
        assert pattern_occurs_in_subsequence(pattern, subsequence)

    @pytest.mark.parametrize('nb_symbols', [3, 4, 5, 6])
    @pytest.mark.parametrize('length_pattern', [4, 8, 12])
    @pytest.mark.parametrize('seed', range(15))
    def test_pattern_occurs_in_subsequence_pattern_does_not_occur(self, nb_symbols, length_pattern, seed):
        np.random.seed(seed)
        pattern = np.random.choice(nb_symbols, length_pattern)
        subsequence = np.random.choice(nb_symbols, np.random.choice([10, 15, 20, 25]))
        for start_pos in range(subsequence.shape[0] - length_pattern + 1):
            if np.array_equal(pattern, subsequence[start_pos:start_pos+length_pattern]):
                assert pattern_occurs_in_subsequence(pattern, subsequence)
                return
        assert not pattern_occurs_in_subsequence(pattern, subsequence)

    def test_windowed_to_observation_embedding_simple(self):
        window_based_embedding = np.array([
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ])
        observation_based_embedding = windowed_to_observation_embedding(window_based_embedding, 3, 1, 12)
        assert np.array_equal(observation_based_embedding, np.array([[1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 9.5, 10]]))

        observation_based_embedding = windowed_to_observation_embedding(window_based_embedding, 3, 1, 13)
        assert np.array_equal(observation_based_embedding, np.array([[1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 9.5, 10, 10]]))

        observation_based_embedding = windowed_to_observation_embedding(window_based_embedding, 5, 1, 14)
        assert np.array_equal(observation_based_embedding, np.array([[1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 8.5, 9, 9.5, 10]]))

        observation_based_embedding = windowed_to_observation_embedding(window_based_embedding, 3, 2, 21)
        assert np.array_equal(observation_based_embedding, np.array([[1, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10]]))

        observation_based_embedding = windowed_to_observation_embedding(window_based_embedding, 5, 2, 23)
        assert np.array_equal(observation_based_embedding, np.array([[1, 1, 1.5, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 9.5, 10, 10]]))

    @pytest.mark.parametrize('window_size', [16, 64])
    @pytest.mark.parametrize('stride', [1, 8])
    @pytest.mark.parametrize('time_series_length', [500, 1000, 2500])
    @pytest.mark.parametrize('seed', range(5))
    def test_windowed_to_observation_embedding(self, window_size, stride, time_series_length, seed):
        np.random.seed(seed)
        nb_subsequences = (time_series_length - window_size) // stride + 1
        window_based_embedding = np.random.uniform(size=(np.random.choice([25, 50, 100]), nb_subsequences))
        observation_based_embedding = windowed_to_observation_embedding(window_based_embedding, window_size, stride, time_series_length)

        nb_segments = ((time_series_length - window_size) // stride) + 1
        start_segments = np.arange(nb_segments) * stride
        end_segments = start_segments + window_size
        end_segments[-1] = time_series_length

        for t in range(time_series_length):
            covering_segments = []
            for i, (start, end) in enumerate(zip(start_segments, end_segments)):
                if start <= t < end:
                    covering_segments.append(i)

            assert observation_based_embedding[:, t] == pytest.approx(np.mean(window_based_embedding[:, covering_segments], axis=1))
