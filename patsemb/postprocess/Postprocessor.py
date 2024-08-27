
import abc
import numpy as np


class Postprocessor(abc.ABC):
    """
    A class for postprocessing the embedding matrix. Postprocessing applies
    an additional transformation on the matrix in order to slightly change
    the values.

    See Also
    --------
    Smoother: Apply temporal smoothing on the embedding matrix.
    """

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y=None) -> 'Postprocessor':
        """
        Fit this postprocessor.

        Parameters
        ----------
        X: np.ndarray of shape (n_patterns, n_samples)
            The embedding matrix to use for fitting this postprocessor.
        y: Ignored
            Is passed for fitting the discretizer, but will typically not be used and
            is only present here for API consistency by convention.

        Returns
        -------
        self: Postprocessor
            Returns the instance itself
        """

    @abc.abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the given pattern-based embedding using this postprocessor.

        Parameters
        ----------
        X: np.ndarray of shape (n_patterns, n_samples)
            The embedding matrix which should be transformed.

        Returns
        -------
        transformed_embedding_matrix: np.ndarray of shape (n_patterns, n_samples)
            The transformed version of the embedding matrix.
        """

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Fit this postprocessor using the given pattern-based embedding, and
        immediately transform it.

        Parameters
        ----------
        X: np.ndarray of shape (n_patterns, n_samples)
            The embedding matrix to use for fitting this postprocessor and which
            should be transformed.
        y: Ignored
            Is passed for fitting the discretizer, but will typically not be used and
            is only present here for API consistency by convention.

        Returns
        -------
        transformed_embedding_matrix: np.ndarray of shape (n_patterns, n_samples)
            The transformed version of the embedding matrix.
        """
        return self.fit(X, y).transform(X)

