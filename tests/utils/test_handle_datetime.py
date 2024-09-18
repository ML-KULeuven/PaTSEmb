
import pytest
import numpy as np
from patsemb.utils import timedelta_to_nb_observations


class TestHandleDatetime:

    @pytest.mark.parametrize('unit', ['s', 'm', 'h', 'D'])
    @pytest.mark.parametrize('ratio', [1, 2, 4, 8])
    def test_timedelta_to_nb_observations(self, unit, ratio):
        timedelta_value = np.timedelta64(ratio, unit)
        timestamps = np.array([np.datetime64('now') + i * np.timedelta64(1, unit) for i in range(1000)])
        nb_observations = timedelta_to_nb_observations(timedelta_value, timestamps)
        assert nb_observations == ratio
        assert isinstance(nb_observations, int)

    def test_timedelta_to_nb_observations_mixed_units(self):
        timedelta_value = np.timedelta64(2, 'h')
        timestamps = np.array([np.datetime64('now') + i * np.timedelta64(10, 'm') for i in range(1000)])
        nb_observations = timedelta_to_nb_observations(timedelta_value, timestamps)
        assert nb_observations == 12
        assert isinstance(nb_observations, int)

    def test_timedelta_to_nb_observations_non_nice_values(self):
        timedelta_value = np.timedelta64(55, 'm')
        timestamps = np.array([np.datetime64('now') + i * np.timedelta64(10, 'm') for i in range(1000)])
        nb_observations = timedelta_to_nb_observations(timedelta_value, timestamps)
        assert nb_observations == 6
        assert isinstance(nb_observations, int)

    def test_timedelta_to_nb_observations_just_not_too_large(self):
        timedelta_value = np.timedelta64(10, 's')
        timestamps = np.array([np.datetime64('now') + i * np.timedelta64(1, 's') for i in range(11)])
        nb_observations = timedelta_to_nb_observations(timedelta_value, timestamps)
        assert nb_observations == 10
        assert isinstance(nb_observations, int)

    def test_timedelta_to_nb_observations_too_large(self):
        timedelta_value = np.timedelta64(1, 'D')
        timestamps = np.array([np.datetime64('now') + i * np.timedelta64(1, 's') for i in range(1000)])
        with pytest.raises(ValueError):
            timedelta_to_nb_observations(timedelta_value, timestamps)
