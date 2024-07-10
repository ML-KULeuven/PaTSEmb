
import pytest
import numpy as np
from typing import List
from patsemb.pattern_mining.SPMF.QCSP import QCSP


@pytest.fixture
def default_miner() -> QCSP:
    return QCSP()


@pytest.fixture
def example_input() -> np.ndarray:
    return np.array([
        [1, 2, 3, 4],
        [4, 3, 2, 1]
    ])


@pytest.fixture
def example_output() -> List[np.array]:
    return [
        np.array([1, 2, 3]),
        np.array([2, 3, 4]),
        np.array([1, 3, 4]),
        np.array([1, 2, 4]),
        np.array([2, 4]),
        np.array([2, 3]),
        np.array([2, 1]),
        np.array([1, 4]),
        np.array([1, 3]),
        np.array([1, 2]),
    ]


class TestNOSEP:

    def test_mining_algorithm_name(self, default_miner: QCSP):
        assert default_miner.mining_algorithm() == 'QCSP'

    @pytest.mark.parametrize('minimum_support', [10, 50, 100])
    @pytest.mark.parametrize('alpha', [1, 2, 3, 4, 5])
    @pytest.mark.parametrize('maximum_length', [8, 12, 16])
    @pytest.mark.parametrize('top_k_patterns', [25, 50, 200])
    def test_hyperparameters(self, minimum_support, alpha, maximum_length, top_k_patterns):
        miner = QCSP(
            minimum_support=minimum_support,
            alpha=alpha,
            maximum_length=maximum_length,
            top_k_patterns=top_k_patterns
        )
        expected_hyperparameters = f'{minimum_support} {alpha} {maximum_length} {top_k_patterns}'
        assert miner.hyperparameters() == expected_hyperparameters

    def test_encode_input_string_example(self, default_miner: QCSP, example_input: np.ndarray):
        encoded = default_miner._encode_input_string(example_input)
        assert encoded == "1 -1 2 -1 3 -1 4 -1 -2\n4 -1 3 -1 2 -1 1 -1 -2"

    @pytest.mark.parametrize('nb_sequences', [5, 10, 50, 100])
    @pytest.mark.parametrize('sequence_length', [5, 7, 9])
    @pytest.mark.parametrize('nb_symbols', [2, 3, 4])
    @pytest.mark.parametrize('seed', [42])
    def test_encode_input_string(self, default_miner: QCSP, nb_sequences, sequence_length, nb_symbols, seed):
        np.random.seed(seed=seed)
        discrete_sequences = np.random.randint(low=0, high=nb_symbols, size=(nb_sequences, sequence_length))
        encoded = default_miner._encode_input_string(discrete_sequences)

        encoded = encoded.replace(' ', '')  # Remove spaces
        encoded = encoded.replace('\n', '')  # Remove new line tokens
        assert encoded.endswith('-2')
        encoded_sequences = encoded[:-2].split('-2')
        assert len(encoded_sequences) == nb_sequences
        for sequence_id, encoded_sequence in enumerate(encoded_sequences):
            assert encoded_sequence.endswith('-1')
            elements = encoded_sequence.strip()[:-2].split('-1')
            assert len(elements) == sequence_length
            for element_id, element in enumerate(elements):
                assert discrete_sequences[sequence_id, element_id] == int(element)

    def test_decode_output_string_example(self, default_miner: QCSP, example_output: List[np.array]):
        output_string = [
            "1 -1 2 -1 3 -1  # SUP: 6   #QCOH: 0,500",
            "2 -1 3 -1 4 -1  # SUP: 6   #QCOH: 0,500",
            "1 -1 3 -1 4 -1  # SUP: 6   #QCOH: 0,500",
            "1 -1 2 -1 4 -1  # SUP: 6   #QCOH: 0,500",
            "2 -1 4 -1  # SUP: 4   #QCOH: 0,500",
            "2 -1 3 -1  # SUP: 4   #QCOH: 0,500",
            "2 -1 1 -1  # SUP: 4   #QCOH: 0,500",
            "1 -1 4 -1  # SUP: 4   #QCOH: 0,500",
            "1 -1 3 -1  # SUP: 4   #QCOH: 0,500",
            "1 -1 2 -1  # SUP: 4   #QCOH: 0,500"
        ]
        patterns = default_miner._decode_output_string(output_string)
        assert len(patterns) == len(example_output)
        for pattern, expected_pattern in zip(patterns, example_output):
            assert np.array_equal(pattern, expected_pattern)

    def test_mine_patterns_example(self, example_input: np.ndarray, example_output: List[np.array]):
        miner = QCSP(
            minimum_support=2,
            alpha=3,
            maximum_length=3,
            top_k_patterns=10
        )
        patterns = miner.mine(example_input)
        assert len(patterns) == len(example_output)
        for pattern, expected_pattern in zip(patterns, example_output):
            assert np.array_equal(pattern, expected_pattern)
