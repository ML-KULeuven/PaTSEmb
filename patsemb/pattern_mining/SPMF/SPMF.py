
import os
import subprocess
import pathlib
import tempfile
import abc
import numpy as np
from typing import List

from patsemb.pattern_mining.PatternMiner import PatternMiner


class SPMF(PatternMiner, abc.ABC):

    def mine(self, discrete_sequences: np.ndarray) -> List[np.array]:
        # Create an input file and write the discrete subsequences to it
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_file:
            input_file_name = tmp_file.name
            tmp_file.write(str.encode(self._encode_input_string(discrete_sequences)))

        # Create an output file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            output_file_name = tmp_file.name

        # Execute the java command to mine the sequences
        command = f"java -jar {pathlib.Path(__file__).parent}/spmf.jar run {self.mining_algorithm()} {input_file_name} {output_file_name} {self.hyper_parameters()}"
        _ = subprocess.run(command, stdout=subprocess.DEVNULL)

        # Read the output file
        with open(output_file_name, 'r') as output_file:
            patterns = self._decode_output_string(output_file.readlines())

        # Clean up the files
        os.remove(input_file_name)
        os.remove(output_file_name)

        # Return the patterns
        return patterns

    @abc.abstractmethod
    def mining_algorithm(self) -> str:
        raise NotImplementedError("Method 'SPMF.mining_algorithm()' should be implemented in the child!")

    @abc.abstractmethod
    def hyper_parameters(self) -> str:
        raise NotImplementedError("Method 'SPMF.hyper_parameters()' should be implemented in the child!")

    @abc.abstractmethod
    def _encode_input_string(self, discrete_sequences: np.ndarray) -> str:
        raise NotImplementedError("Method 'SPMF._encode_input_string()' should be implemented in the child!")

    @abc.abstractmethod
    def _decode_output_string(self, output_lines: List[str]) -> List[np.array]:
        raise NotImplementedError("Method 'SPMF._decode_output_string()' should be implemented in the child!")
