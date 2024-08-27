import numpy as np
from typing import List

from patsemb.pattern_mining.SPMF import SPMF


class QCSP(SPMF):
    """
    Mine frequent, sequential patterns using the QCSP algorithm.
    """

    def __init__(self,
                 minimum_support: int = 3,
                 alpha: int = 3,
                 maximum_length: int = 4,
                 top_k_patterns: int = 25):  # TODO check default parameters wih paper
        self.minimum_support: int = minimum_support
        self.alpha: int = alpha
        self.maximum_length: int = maximum_length
        self.top_k_patterns: int = top_k_patterns

    def mining_algorithm(self) -> str:
        return 'QCSP'

    def hyperparameters(self) -> str:
        return f'{self.minimum_support} {self.alpha} {self.maximum_length} {self.top_k_patterns}'

    def _encode_input_string(self, discrete_sequences: np.ndarray) -> str:
        return ' -1 -2\n'.join([' -1 '.join(pattern.astype(str)) for pattern in discrete_sequences]) + ' -1 -2'

    def _decode_output_string(self, output_lines: List[str]) -> List[np.array]:
        return [np.array(output_line.split(' -1 ')[:-1], dtype=int) for output_line in output_lines]
