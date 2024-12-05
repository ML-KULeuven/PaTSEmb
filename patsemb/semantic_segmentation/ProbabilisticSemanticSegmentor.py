
import abc
import numpy as np


class ProbabilisticSemanticSegmentor(abc.ABC):
    """
    Learn a probabilistic semantic segmentation over the pattern-based
    embedding. This enables to learn gradual transitions over the semantic
    segmentation as intervals where the probability of one semantic segment
    increases while the probability of another semantic segment decreases.

    Because segment probabilities are predicted, this class uses the fit-predict_proba
    interface (including a ``fit_predict_proba`` method) to make predictions.

    See Also
    --------
    LogisticRegressionSegmentor: predict semantic segments using logistic regression.
    """

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y=None) -> 'ProbabilisticSemanticSegmentor':
        """
        Fit this probabilistic semantic segmentor.

        Parameters
        ----------
        X: np.ndarray of shape (n_patterns, n_samples)
            The embedding matrix to use for fitting this probabilistic semantic segmentor.
        y: array-like, default=None
            Ground-truth information.

        Returns
        -------
        self: ProbabilisticSemanticSegmentor
            Returns the instance itself
        """

    @abc.abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the probabilistic semantic segment probabilities, based on
        the given pattern-based embedding.

        Parameters
        ----------
        X: np.ndarray of shape (n_patterns, n_samples)
            The embedding matrix which should be transformed.

        Returns
        -------
        segment_probabilities: np.ndarray of shape (n_samples, n_segments)
            The predicted semantic segment probabilities.
        """

    def fit_predict_proba(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Fit this postprocessor using the given pattern-based embedding, and
        immediately transform it.

        Parameters
        ----------
        X: np.ndarray of shape (n_patterns, n_samples)
            The embedding matrix to use for fitting this probabilistic semantic segmentor.
        y: array-like, default=None
            Ground-truth information.

        Returns
        -------
        segment_probabilities: np.ndarray of shape (n_samples, n_segments)
            The predicted semantic segment probabilities.
        """
        return self.fit(X, y).predict_proba(X)
