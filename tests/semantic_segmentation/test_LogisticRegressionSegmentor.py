import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from patsemb.semantic_segmentation import LogisticRegressionSegmentor
from patsemb.pattern_based_embedding import PatternBasedEmbedder


@pytest.fixture
def pattern_based_embedding() -> np.ndarray:
    univariate_time_series = np.sin(np.arange(0, 50, 0.05)) + np.random.normal(0, 0.25, 1000)
    return PatternBasedEmbedder().fit_transform(univariate_time_series)


class TestLogisticRegressionSegmentor:

    def test_initialization_n_segments(self):
        clf = LogisticRegressionSegmentor()
        assert clf.n_segments == [2, 3, 4, 5, 6, 7, 8, 9]

        clf = LogisticRegressionSegmentor(n_segments=[2, 3, 4, 5])
        assert clf.n_segments == [2, 3, 4, 5]

        clf = LogisticRegressionSegmentor(n_segments=4)
        assert clf.n_segments == [4]

    def test_initialization_n_jobs(self):
        clf = LogisticRegressionSegmentor()
        assert clf.n_jobs == 1

        clf = LogisticRegressionSegmentor(n_jobs=4)
        assert clf.n_jobs == 4

    def test_initialization_kwargs(self):
        clf = LogisticRegressionSegmentor(penalty='l2', tol=1e-4, init='random', max_iter=50)
        assert 'tol' in clf.k_means_kwargs
        assert 'init' in clf.k_means_kwargs
        assert 'max_iter' in clf.k_means_kwargs
        assert len(clf.k_means_kwargs) == 3

        assert 'penalty' in clf.logistic_regression_kwargs
        assert 'tol' in clf.logistic_regression_kwargs
        assert 'max_iter' in clf.logistic_regression_kwargs
        assert len(clf.logistic_regression_kwargs) == 3

    def test_initialization_n_clusters(self):
        with pytest.raises(TypeError):
            LogisticRegressionSegmentor(n_clusters=5)

    def test_initialization_additional_args(self):
        with pytest.raises(TypeError):
            LogisticRegressionSegmentor(something_invalid=0)

    def test_fit(self):
        univariate_time_series = np.sin(np.arange(0, 50, 0.05)) + np.random.normal(0, 0.25, 1000)
        pattern_based_embedding = PatternBasedEmbedder().fit_transform(univariate_time_series)
        clf = LogisticRegressionSegmentor()
        assert clf.fit(pattern_based_embedding) == clf

    def test_predict_proba(self):
        univariate_time_series = np.sin(np.arange(0, 50, 0.05)) + np.random.normal(0, 0.25, 1000)
        pattern_based_embedding = PatternBasedEmbedder().fit_transform(univariate_time_series)
        clf = LogisticRegressionSegmentor()
        clf.fit(pattern_based_embedding)
        pred = clf.predict_proba(pattern_based_embedding)
        assert pred.shape[0] == pattern_based_embedding.shape[1]

    def test_fit_predict_proba(self):
        univariate_time_series = np.sin(np.arange(0, 50, 0.05)) + np.random.normal(0, 0.25, 1000)
        pattern_based_embedding = PatternBasedEmbedder().fit_transform(univariate_time_series)
        pred = LogisticRegressionSegmentor().fit_predict_proba(pattern_based_embedding)
        assert pred.shape[0] == pattern_based_embedding.shape[1]

    # def test_fit_predict_proba_one_n_segment(self):
    #     univariate_time_series = np.sin(np.arange(0, 50, 0.05)) + np.random.normal(0, 0.25, 1000)
    #     pattern_based_embedding = PatternBasedEmbedder().fit_transform(univariate_time_series)
    #     pred = LogisticRegressionSegmentor(n_segments=3).fit_predict_proba(pattern_based_embedding)
    #     assert pred.shape == (pattern_based_embedding.shape[1], 3)
    #
    # def test_fit_predict_proba_multiple_jobs(self):
    #     univariate_time_series = np.sin(np.arange(0, 50, 0.05)) + np.random.normal(0, 0.25, 1000)
    #     pattern_based_embedding = PatternBasedEmbedder().fit_transform(univariate_time_series)
    #     pred = LogisticRegressionSegmentor(n_jobs=4).fit_predict_proba(pattern_based_embedding)
    #     assert pred.shape[0] == pattern_based_embedding.shape[1]
    #
    # def test_predict_proba_not_fitted(self):
    #     univariate_time_series = np.sin(np.arange(0, 50, 0.05)) + np.random.normal(0, 0.25, 1000)
    #     pattern_based_embedding = PatternBasedEmbedder().fit_transform(univariate_time_series)
    #     with pytest.raises(NotFittedError):
    #         LogisticRegressionSegmentor().predict_proba(pattern_based_embedding)
