import numpy as np


def timedelta_to_nb_observations(timedelta_value: np.timedelta64, time_stamps: np.array) -> int:
    """
    A utility method to convert time delta values to a number of observations. This method
    will search for the minimal number of observations such that they cover at least the
    time period given by ``timedelta_value`` in the provided ``time_stamps``. This method
    assumes that the provided ``time_stamps`` are sorted, with the earliest time stamp
    at the first position and the latest time stamp at the last position. The given
    time stamps should be regularly sampled.

    Parameters
    ----------
    timedelta_value: np.timedelta64
        The time interval that must be covered by the computed number of observations.
    time_stamps: np.array of np.datetime64
        The time stamps corresponding to the observation times of a time series.

    Returns
    -------
    nb_observations: int
        The minimal number of consecutive observations that cover the given timedelta
        value within the given time stamps.
    """
    if time_stamps[-1] - time_stamps[0] < timedelta_value:
        raise ValueError("The given timedelta value is too large for the given time stamps!")

    nb_observations = 1
    while time_stamps[nb_observations] - time_stamps[0] < timedelta_value:
        nb_observations += 1
    return nb_observations
