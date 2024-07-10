
import pytest
import numpy as np
from typing import List
from patsemb.pattern_mining.SPMF.NOSEP import NOSEP


@pytest.fixture
def default_miner() -> NOSEP:
    return NOSEP()


@pytest.fixture
def example_input() -> np.ndarray:
    return np.array([1, 1, 7, 20, 1, 3, 7, 1, 3, 7, 3, 1, 20, 3, 20, 1]).reshape(1, -1)


@pytest.fixture
def example_output() -> List[np.array]:
    return [
        np.array([1]),
        np.array([3]),
        np.array([7]),
        np.array([20]),
        np.array([1, 1]),
        np.array([1, 3]),
        np.array([1, 7]),
        np.array([3, 1]),
        np.array([3, 3]),
        np.array([7, 1]),
        np.array([7, 3]),
        np.array([1, 1, 7]),
        np.array([1, 3, 1]),
        np.array([1, 7, 1]),
        np.array([1, 7, 3]),
        np.array([7, 1, 3]),
        np.array([7, 3, 3]),
        np.array([1, 1, 7, 1]),
        np.array([1, 1, 7, 3]),
        np.array([1, 7, 1, 3]),
        np.array([1, 7, 3, 3]),
        np.array([7, 1, 3, 1]),
        np.array([1, 1, 7, 1, 3]),
        np.array([1, 1, 7, 3, 3]),
        np.array([1, 7, 1, 3, 1]),
        np.array([1, 1, 7, 1, 3, 1])
    ]


class TestNOSEP:

    def test_mining_algorithm_name(self, default_miner: NOSEP):
        assert default_miner.mining_algorithm() == 'NOSEP'

    @pytest.mark.parametrize('minimum_length', [1, 2, 3, 4])
    @pytest.mark.parametrize('maximum_length', [5, 7, 9, 11])
    @pytest.mark.parametrize('minimum_gap', [0, 1])
    @pytest.mark.parametrize('maximum_gap', [1, 2, 3, 4, 5])
    @pytest.mark.parametrize('minimum_support', [10, 50, 100])
    def test_hyperparameters(self, minimum_length, maximum_length, minimum_gap, maximum_gap, minimum_support):
        miner = NOSEP(
            minimum_length=minimum_length,
            maximum_length=maximum_length,
            minimum_gap=minimum_gap,
            maximum_gap=maximum_gap,
            minimum_support=minimum_support
        )
        expected_hyperparameters = f'{minimum_length} {maximum_length} {minimum_gap} {maximum_gap} {minimum_support}'
        assert miner.hyperparameters() == expected_hyperparameters

    def test_encode_input_string_example(self, default_miner: NOSEP, example_input: np.ndarray):
        encoded = default_miner._encode_input_string(example_input)
        assert encoded == "1 -1 1 -1 7 -1 20 -1 1 -1 3 -1 7 -1 1 -1 3 -1 7 -1 3 -1 1 -1 20 -1 3 -1 20 -1 1 -1 -2"

    @pytest.mark.parametrize('nb_sequences', [5, 10, 50, 100])
    @pytest.mark.parametrize('sequence_length', [5, 7, 9])
    @pytest.mark.parametrize('nb_symbols', [2, 3, 4])
    @pytest.mark.parametrize('seed', [42])
    def test_encode_input_string(self, default_miner: NOSEP, nb_sequences, sequence_length, nb_symbols, seed):
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

    def test_decode_output_string_example(self, default_miner: NOSEP, example_output: List[np.array]):
        output_string = [
            "1 -1 #SUP: 6",
            "3 -1 #SUP: 4",
            "7 -1 #SUP: 3",
            "20 -1 #SUP: 3",
            "1 -1 1 -1 #SUP: 3",
            "1 -1 3 -1 #SUP: 3",
            "1 -1 7 -1 #SUP: 3",
            "3 -1 1 -1 #SUP: 3",
            "3 -1 3 -1 #SUP: 3",
            "7 -1 1 -1 #SUP: 3",
            "7 -1 3 -1 #SUP: 3",
            "1 -1 1 -1 7 -1 #SUP: 3",
            "1 -1 3 -1 1 -1 #SUP: 3",
            "1 -1 7 -1 1 -1 #SUP: 3",
            "1 -1 7 -1 3 -1 #SUP: 3",
            "7 -1 1 -1 3 -1 #SUP: 3",
            "7 -1 3 -1 3 -1 #SUP: 3",
            "1 -1 1 -1 7 -1 1 -1 #SUP: 3",
            "1 -1 1 -1 7 -1 3 -1 #SUP: 3",
            "1 -1 7 -1 1 -1 3 -1 #SUP: 3",
            "1 -1 7 -1 3 -1 3 -1 #SUP: 3",
            "7 -1 1 -1 3 -1 1 -1 #SUP: 3",
            "1 -1 1 -1 7 -1 1 -1 3 -1 #SUP: 3",
            "1 -1 1 -1 7 -1 3 -1 3 -1 #SUP: 3",
            "1 -1 7 -1 1 -1 3 -1 1 -1 #SUP: 3",
            "1 -1 1 -1 7 -1 1 -1 3 -1 1 -1 #SUP: 3"
        ]
        patterns = default_miner._decode_output_string(output_string)
        assert len(patterns) == len(example_output)
        for pattern, expected_pattern in zip(patterns, example_output):
            assert np.array_equal(pattern, expected_pattern)

    def test_mine_patterns_example(self, example_input: np.ndarray, example_output: List[np.array]):
        miner = NOSEP(
            minimum_length=1,
            maximum_length=20,
            minimum_gap=0,
            maximum_gap=2,
            minimum_support=3
        )
        patterns = miner.mine(example_input)
        assert len(patterns) == len(example_output)
        for pattern, expected_pattern in zip(patterns, example_output):
            assert np.array_equal(pattern, expected_pattern)
