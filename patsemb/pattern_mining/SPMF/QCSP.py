import numpy as np
from typing import List

from patsemb.pattern_mining.SPMF import SPMF


class QCSP(SPMF):

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

    def hyper_parameters(self) -> str:
        return f'{self.minimum_support} {self.alpha} {self.maximum_length} {self.top_k_patterns}'

    def _encode_input_string(self, discrete_sequences: np.ndarray) -> str:
        return ' -1 -2\n'.join([' -1 '.join(pattern.astype(str)) for pattern in discrete_sequences]) + ' -1 -2'

    def _decode_output_string(self, output_lines: List[str]) -> List[np.array]:
        return [np.array(output_line.split(' -1 ')[:-1], dtype=int) for output_line in output_lines]


def main():
    np.random.seed(42)
    time_series = np.random.normal(size=1000)

    from patsemb.discretization import SAXDiscretizer
    discretizer = SAXDiscretizer(
        alphabet_size=5,
        word_size=10,
        window_size=25,
        stride=25,
        discretize_within='time_series'
    )
    discrete_sequences = discretizer.fit_transform(time_series)

    spmf = QCSP(
        minimum_support=3,
        alpha=3,
        maximum_length=4,
        top_k_patterns=25
    )
    patterns = spmf.mine(discrete_sequences)
    for pattern in patterns:
        print(pattern)


if __name__ == '__main__':
    main()