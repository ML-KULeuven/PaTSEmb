
import inspect
import multiprocessing
import numpy as np
from typing import Union, List

from sklearn.exceptions import NotFittedError
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression

from patsemb.semantic_segmentation.ProbabilisticSemanticSegmentor import ProbabilisticSemanticSegmentor


class LogisticRegressionSegmentor(ProbabilisticSemanticSegmentor):
    """
    Segments the pattern-based embedding using Logistic Regression [carpentier2024pattern]_.

    First, a KMeans clustering model is fitted on the embedding, which will
    provide a discrete clustering (i.e., every observation in the time series
    will be assigned a discrete cluster label). The number of clusters `K` is
    decided based on the silhouette method. The discrete clustering give an
    initial indication of when the semantic segments occur.

    Second, the discrete clustering is fed to a logistic regression model. This
    model learns to which segment each time point of the pattern-based embedding
    belongs. Because logistic regression is a probabilistic model, we retrieve
    the probabilities of a given observation belong to a semantic segment,
    thereby obtaining a probabilistic segmentation.

    Parameters
    ----------
    n_segments: int or list of int, default=[2, 3, 4, 5, 6, 7, 8, 9]
        The number of segments. If a list of integers is passed, a clustering
        will be made for each value, and the best clustering is selected using
        the silhouette score.
    n_jobs: int, default=1
        The number of jobs to use for computing the multiple clusterings. Has
        no effect if ``n_segments`` is an integer.
    **kwargs:
        Additional arguments to be passed to either ``KMeans`` clutering or
        ``LogisticRegression`` (both using Sklearn implementation). This class
        automatically infers which parameters can be passed to either object
        using the ``inspect`` module. If a parameter is valid for both models
        (e.g., ``max_iter``), then it will be passed to both. If an additional
        argument is given, which is not valid for KMeans nor for LogisticRegression,
        a TypeError will be thrown.

        A TypeError will also be raised if ``n_clusters`` is passed to this
        object - even though it is valid for ``KMeans`` - because this parameter
        will be set based on ``n_segments``.

    Attributes
    ----------
    k_means_kwargs: dict
        The arguments to pass to SKlearn KMeans.
    logistic_regression_kwargs: dict
        The arguments to pass to SKlearn LogisticRegression.
    logistic_regression_: LogisticRegression
        The fitted SKlearn Logistic Regression model.

    References
    ----------
    .. [carpentier2024pattern] Carpentier, Louis, Feremans, Len, Meert, Wannes, Verbeke, Mathias.
       "Pattern-based Time Series Semantic Segmentation with Gradual State Transitions." Proceedings
       of the 2024 SIAM International Conference on Data Mining (SDM). Society for Industrial and
       Applied Mathematics, 2024, doi: `10.1137/1.9781611978032.36 <https://doi.org/10.1137/1.9781611978032.36>`_.
    """
    n_segments: Union[int, List[int]]
    n_jobs: int
    kwargs: dict

    k_means_kwargs: dict
    logistic_regression_kwargs: dict

    logistic_regression_: LogisticRegression

    def __init__(self,
                 n_segments: Union[List[int], int] = None,
                 n_jobs: int = 1,
                 **kwargs):

        self.n_segments: List[int] = \
            list(range(2, 10)) if n_segments is None else \
            [n_segments] if isinstance(n_segments, int) else \
            n_segments
        self.n_jobs = n_jobs
        self.kwargs = kwargs

        # Separate the kwargs
        self.k_means_kwargs = {key: value for key, value in kwargs.items() if key in inspect.signature(KMeans).parameters}
        self.logistic_regression_kwargs = {key: value for key, value in kwargs.items() if key in inspect.signature(LogisticRegression).parameters}

        if 'n_clusters' in self.k_means_kwargs:
            raise TypeError("Parameter 'n_clusters' should not be passed!")

        # Check if invalid arguments were given
        valid_kwargs = dict(self.k_means_kwargs, **self.logistic_regression_kwargs)
        if len(valid_kwargs) != len(kwargs):
            invalid_kwargs = [arg for arg in kwargs.keys() if arg not in valid_kwargs]
            raise TypeError(f"Parameters were given that do not belong to K-Means or Logistic Regression: {invalid_kwargs}")

    def fit(self, X: np.ndarray, y=None) -> 'ProbabilisticSemanticSegmentor':

        # If there is only one value for n_segments given, we can simply compute the clustering
        if len(self.n_segments) == 1:
            clustering = KMeans(n_clusters=self.n_segments[0], **self.k_means_kwargs).fit_predict(X.T)

        # Otherwise, use parallelization and select the best clustering
        else:

            # Compute clusters with different number of segments
            args = [(X.T, n_segments) for n_segments in self.n_segments]
            if self.n_jobs > 1:
                with multiprocessing.Pool(self.n_jobs) as pool:
                    pool_results = pool.starmap(self._compute_kmeans_segmentation, args)
            else:
                pool_results = [self._compute_kmeans_segmentation(*arg) for arg in args]

            # Identify the best cluster with maximum silhouette score
            index_largest_silhouette_score = np.argmax([silhouette_avg for silhouette_avg, *_ in pool_results])
            clustering = pool_results[index_largest_silhouette_score][1]

        # Fit the logistic regression model
        self.logistic_regression_ = LogisticRegression(**self.logistic_regression_kwargs)
        self.logistic_regression_.fit(X.T, clustering)

        # Return self
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, 'logistic_regression_'):
            raise NotFittedError('Call the fit method before predicting!')
        return self.logistic_regression_.predict_proba(X.T)

    def _compute_kmeans_segmentation(self, X: np.ndarray, n_segments: int):
        # Cluster the embedding
        k_means = KMeans(n_clusters=n_segments, **self.k_means_kwargs)
        segmentation = k_means.fit_predict(X)

        # Compute silhouette score
        if len(set(segmentation)) != n_segments:
            silhouette_avg = -1
        else:
            n = X.shape[0]
            sample_size = n if n < 2000 else 2000 + int(0.1 * (n - 2000))
            silhouette_avg = silhouette_score(X, segmentation, sample_size=sample_size)

        return silhouette_avg, segmentation
