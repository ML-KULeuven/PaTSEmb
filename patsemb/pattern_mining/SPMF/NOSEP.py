
import numpy as np
from typing import List

from patsemb.pattern_mining.SPMF import SPMF


class NOSEP(SPMF):

    def __init__(self,
                 minimum_length: int = 1,
                 maximum_length: int = 20,
                 minimum_gap: int = 0,
                 maximum_gap: int = 2,
                 minimum_support: int = 10):  # TODO check default parameters wih paper
        self.minimum_length: int = minimum_length
        self.maximum_length: int = maximum_length
        self.minimum_gap: int = minimum_gap
        self.maximum_gap: int = maximum_gap
        self.minimum_support: int = minimum_support

    def mining_algorithm(self) -> str:
        return 'NOSEP'

    def hyper_parameters(self) -> str:
        return f'{self.minimum_length} {self.maximum_length} {self.minimum_gap} {self.maximum_gap} {self.minimum_support}'

    def _encode_input_string(self, discrete_sequences: List[np.array]) -> str:
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

    spmf = NOSEP(
        minimum_length=1,
        maximum_length=20,
        minimum_gap=0,
        maximum_gap=2,
        minimum_support=10
    )
    patterns = spmf.mine(discrete_sequences)
    for pattern in patterns:
        print(pattern)


if __name__ == '__main__':
    main()
