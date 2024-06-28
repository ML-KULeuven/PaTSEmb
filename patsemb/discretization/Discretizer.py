
import abc
import numpy as np
from typing import List, Union


class Discretizer(abc.ABC):

    def __init__(self, window_size: int, stride: int):
        self.window_size: int = window_size
        self.stride: int = stride

    @abc.abstractmethod
    def fit(self, dataset: Union[np.ndarray, List[np.ndarray]], y=None) -> 'Discretizer':
        raise NotImplementedError("Method 'Discretizer.fit()' should be implemented in the child!")

    @abc.abstractmethod
    def transform(self, dataset: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        raise NotImplementedError("Method 'Discretizer.transform()' should be implemented in the child!")

    def fit_transform(self, dataset: Union[np.ndarray, List[np.ndarray]], y=None) -> np.ndarray:
        return self.fit(dataset, y).transform(dataset)
