
import numpy as np
import numba as nb

from patsemb.postprocess.Postprocessor import Postprocessor


class Smoother(Postprocessor):
    """
    Apply temporal smoothing on the embedding matrix. Temporal smoothing
    will transform each column in the embedding matrix by taking the weighted
    average of that column and the neighboring columns. This ensures that the
    consecutive time steps have more similar values.

    Specifically, the transformation of column i in embedding matrix E with
    normalized weights W equals:

        E[:, i] = ... + W[1] * E[:, i-1] + W[0] * E[:, i] + W[1] * E[:, i+1] + ...

    Parameters
    ----------
    nb_iterations: int, default=1
        The number of times smoothing will be applied. This value must be at
        least 1.
    weights: np.array of shape (size_neighborhood,)
        The weights used for aggregating the columns. The first weight corresponds
        to the weight of the current column, the other weights correspond to the
        weights of neighboring columns. The weights are interpreted to be relative
        and will be normalized during smoothing. At least two weights must be provided.
        None of the weights can be negative.
    """

    def __init__(self, nb_iterations: int = 1, weights: np.array = None):
        self.nb_iterations: int = nb_iterations
        self.weights: np.array = np.array([2, 1]) if weights is None else weights

        # Check input
        if self.nb_iterations < 1:
            raise Exception("The 'nb_iterations' parameter in Smoother should be 1 or larger to perform "
                            "at least one iteration!")
        if self.weights.shape[0] < 2:
            raise Exception("The 'weights' parameter in Smoother should have at least 2 weights in order to "
                            "take the neighbouring time steps into account!")
        if self.weights.min() < 0:
            raise Exception("All weights in 'Smoother' should be larger than 0!")

    def fit(self, X: np.ndarray, y=None) -> 'Smoother':
        """
        Fit this Smoother. For smoothing there is no fitting necessary.

        Parameters
        ----------
        X: np.ndarray of shape (n_patterns, n_samples)
            The embedding matrix to use for fitting this postprocessor.
        y: Ignored
            Is passed for fitting the discretizer, but will typically not be used and
            is only present here for API consistency by convention.

        Returns
        -------
        self: Smoother
            Returns the instance itself
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Smooth the given embedding matrix.

        Parameters
        ----------
        X: np.ndarray of shape (n_patterns, n_samples)
            The embedding matrix on which the smoothing should be applied.

        Returns
        -------
        smoothed_embedding_matrix: np.ndarray of shape (n_patterns, n_samples)
            The smoothed embedding matrix.
        """
        return smoothing(X, self.nb_iterations, self.weights)


@nb.njit(fastmath=True)
def smoothing(embedding_matrix: np.ndarray, nb_iterations: int, weights: np.array) -> np.ndarray:
    """
    Method to effectively smooth the embedding matrix.
    """
    # col[i] = ... + weight[1] * col_prev[i-1] + weights[0] * col_prev[0] + weights[1] * col_prev[i+1] + ...

    # Set the weights correctly
    weights = weights / (np.sum(weights) + np.sum(weights[1:]))

    # [..., weight[1], weights[0], weights[1], ..]
    weights_expanded = np.empty(shape=2 * weights.shape[0] - 1)
    weights_expanded[:weights.shape[0]] = weights[::-1]
    weights_expanded[weights.shape[0]:] = weights[1:]

    # Repeat the same process for the given number of iterations
    for _ in range(nb_iterations):
        # Initialize the next iteration
        next_iteration = np.zeros_like(embedding_matrix)

        # The first few and last few columns
        for i in range(weights.shape[0] - 1):
            adjusted_weights = weights_expanded[weights.shape[0] - i - 1:] / weights_expanded[weights.shape[0] - i - 1:].sum()
            for j in range(adjusted_weights.shape[0]):
                next_iteration[:, i] += embedding_matrix[:, j] * adjusted_weights[j]
                next_iteration[:, -i - 1] += embedding_matrix[:, -j - 1] * adjusted_weights[-j]

        # Center columns
        for i in range(weights.shape[0] - 1, embedding_matrix.shape[1] - weights.shape[0] + 1):
            for j in range(-weights.shape[0] + 1, weights.shape[0]):
                next_iteration[:, i] += embedding_matrix[:, i + j] * weights[abs(j)]

        # Update the matrix
        embedding_matrix = next_iteration.copy()

    return embedding_matrix
