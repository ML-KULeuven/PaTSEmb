
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
    def apply(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Apply this postprocessor on the given embedding matrix.

        Parameters
        ----------
        X: np.ndarray of shape (n_patterns, n_samples)
            The embedding matrix on which the posprocessing should
            be applied.
        y: Ignored
            Is passed for fitting the discretizer, but will typically not be used and
            is only present here for API consistency by convention.

        Returns
        -------
        transformed_embedding_matrix: np.ndarray of shape (n_patterns, n_samples)
            The transformed version of the embedding matrix.
        """
