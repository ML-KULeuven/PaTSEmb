import abc
import numpy as np
from typing import List


class PatternMiner(abc.ABC):

    @abc.abstractmethod
    def mine(self, discrete_sequences: np.ndarray) -> List[np.array]:
        raise NotImplementedError("Method 'PatternMiner.mine()' should be implemented in the child!")
