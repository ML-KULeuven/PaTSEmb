
import pytest
import numpy as np
from patsemb.postprocess.Smoother import smoothing, Smoother


class TestSmoother:

    @pytest.mark.parametrize('seed', range(10))
    def test_initialize(self, seed):
        np.random.seed(seed)
        nb_iterations = np.random.randint(1, 100)
        weights = np.random.random(size=np.random.randint(2, 10))
        smoother = Smoother(nb_iterations, weights)
        assert smoother.nb_iterations == nb_iterations
        assert np.array_equal(smoother.weights, weights)

    def test_initialize_default(self,):
        smoother = Smoother()
        assert smoother.nb_iterations == 1
        assert np.array_equal(smoother.weights, np.array([2, 1]))

    def test_initialize_invalid_nb_iterations(self):
        try:
            for i in range(1, 10):
                Smoother(nb_iterations=i)
        except Exception:
            pytest.fail("An exception was thrown while this shouldn't occur!")

        with pytest.raises(Exception):
            Smoother(nb_iterations=0)

        with pytest.raises(Exception):
            Smoother(nb_iterations=-1)

    @pytest.mark.parametrize('seed', range(1))
    def test_initialize_invalid_nb_weights(self, seed):
        np.random.seed(seed)
        try:
            for i in range(1, 10):
                Smoother(weights=np.random.random(size=np.random.randint(2, 10)))
        except Exception:
            pytest.fail("An exception was thrown while this shouldn't occur!")

        with pytest.raises(Exception):
            Smoother(weights=np.array([]))  # Empty

        with pytest.raises(Exception):
            Smoother(weights=np.array([1]))  # Only one element

    @pytest.mark.parametrize('seed', range(3))
    def test_initialize_invalid_weight_values(self, seed):
        np.random.seed(seed)
        try:
            for i in range(1, 10):
                Smoother(weights=np.random.random(size=np.random.randint(2, 10)))
        except Exception:
            pytest.fail("An exception was thrown while this shouldn't occur!")

        weights = np.random.random(size=np.random.randint(5, 10))
        for i in range(1, weights.shape[0]):
            indices_negative = np.random.choice(weights.shape[0], size=i, replace=False)
            weights_tmp = weights.copy()
            weights_tmp[indices_negative] *= -1
            assert np.sum(weights_tmp < 0) == i
            with pytest.raises(Exception):
                Smoother(weights=weights_tmp)  # Empty

    def test_example1(self):
        matrix = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=float)
        weights = np.array([4, 2, 1])
        smoothed = smoothing(matrix, 1, weights)
        assert smoothed == pytest.approx(np.array([[2/7, 4/9, 0.2, 0.1, 0, 0, 0, 0.1, 0.2, 0.4, 0.2, 0.1, 0, 0, 0, 0.1, 2/9, 4/7]]))

    def test_example1_class(self):
        matrix = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=float)
        smoother = Smoother(1, np.array([4, 2, 1]))
        smoothed = smoother.fit_transform(matrix)
        assert smoothed == pytest.approx(np.array([[2 / 7, 4 / 9, 0.2, 0.1, 0, 0, 0, 0.1, 0.2, 0.4, 0.2, 0.1, 0, 0, 0, 0.1, 2 / 9, 4 / 7]]))

    def test_example2(self):
        matrix = np.array([[0, 1, 0, 1, 1, 0, 0, 0, 1, 1]], dtype=float)
        weights = np.array([2, 1])
        smoothed = smoothing(matrix, 1, weights)
        assert smoothed == pytest.approx(np.array([[1/3, 2/4, 2/4, 3/4, 3/4, 1/4, 0, 1/4, 3/4, 1]]))

    def test_example2_class(self):
        matrix = np.array([[0, 1, 0, 1, 1, 0, 0, 0, 1, 1]], dtype=float)
        smoother = Smoother(1, np.array([2, 1]))
        smoothed = smoother.fit_transform(matrix)
        assert smoothed == pytest.approx(np.array([[1/3, 2/4, 2/4, 3/4, 3/4, 1/4, 0, 1/4, 3/4, 1]]))

    def test_example3(self):
        matrix = np.array([[0, 1, 0, 1, 1, 0, 0, 0, 1, 1]], dtype=float)
        weights = np.array([2, 1])
        smoothed = smoothing(matrix, 2, weights)
        assert smoothed == pytest.approx(np.array([[14/36, 22/48, 9/16, 11/16, 10/16, 5/16, 2/16, 5/16, 11/16, 11/12]]))

    def test_example3_class(self):
        matrix = np.array([[0, 1, 0, 1, 1, 0, 0, 0, 1, 1]], dtype=float)
        smoother = Smoother(2, np.array([2, 1]))
        smoothed = smoother.fit_transform(matrix)
        assert smoothed == pytest.approx(np.array([[14/36, 22/48, 9/16, 11/16, 10/16, 5/16, 2/16, 5/16, 11/16, 11/12]]))
