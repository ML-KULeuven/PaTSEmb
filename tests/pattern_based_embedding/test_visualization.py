
import pytest
import numpy as np
from patsemb.pattern_based_embedding import PatternBasedEmbedder, plot_time_series_and_embedding


def generate_random_univariate_time_series() -> np.array:
    return np.random.rand(np.random.choice([500, 1000, 5000]))


@pytest.fixture(scope="class")  # This ensures the processing is done once per test class
def time_series_and_embedding():
    np.random.seed(0)
    time_series = np.random.rand(np.random.choice([500, 1000, 5000]))
    embedder = PatternBasedEmbedder()
    embedding = embedder.fit_transform(time_series)
    return time_series, embedding


class TestPlotTimeSeriesAndEmbedding:

    def test_default(self, time_series_and_embedding):
        time_series, embedding = time_series_and_embedding
        fig = plot_time_series_and_embedding(time_series, embedding)
        assert len(fig.get_axes()[1].get_xticks()) == 0
        assert len(fig.get_axes()[1].get_yticks()) == embedding.shape[0]

    def test_provided_time_stamps(self, time_series_and_embedding):
        time_series, embedding = time_series_and_embedding
        fig = plot_time_series_and_embedding(time_series, embedding, time_stamps=np.arange(time_series.shape[0]))
        assert len(fig.get_axes()[1].get_xticks()) > 0  # You don't know exactly how many, but there should be more than 0
        assert len(fig.get_axes()[1].get_yticks()) == embedding.shape[0]

    def test_provided_time_stamps_and_dont_show_pattern_ids(self, time_series_and_embedding):
        time_series, embedding = time_series_and_embedding
        fig = plot_time_series_and_embedding(time_series, embedding, time_stamps=np.arange(time_series.shape[0]), show_pattern_ids=False)
        assert len(fig.get_axes()[1].get_xticks()) > 0
        assert len(fig.get_axes()[1].get_yticks()) == 0

    def test_provided_dont_show_pattern_ids(self, time_series_and_embedding):
        time_series, embedding = time_series_and_embedding
        fig = plot_time_series_and_embedding(time_series, embedding, show_pattern_ids=False)
        assert len(fig.get_axes()[1].get_xticks()) == 0  # You don't know exactly how many, but there should be more than 0
        assert len(fig.get_axes()[1].get_yticks()) == 0
