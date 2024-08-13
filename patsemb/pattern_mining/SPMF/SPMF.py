
import os
import subprocess
import pathlib
import tempfile
import abc
import numpy as np
from typing import List

from patsemb.pattern_mining.PatternMiner import PatternMiner


class SPMF(PatternMiner, abc.ABC):
    """
    Mine frequent sequential patterns in the discrete representation
    of some time series. This is based on the SPMF-library, which is
    a package consisting of many sequential pattern mining algorithms.
    This class serves as a wrapper approach for the jar-file with the
    source code for the algorithms.

    See Also
    --------
    QCSP: Quantile-based Cohesive Sequential Patterns
    NOSEP: Nonoverlapping Sequence Pattern Mining With Gap Constraints

    References
    ----------
    .. Fournier-Viger, P., Lin, C.W., Gomariz, A., Gueniche, T., Soltani,
       A., Deng, Z., Lam, H. T. (2016). The SPMF Open-Source Data Mining
       Library Version 2. Proc. 19th European Conference on Principles of
       Data Mining and Knowledge Discovery (PKDD 2016) Part III.
       https://doi.org/10.1007/978-3-319-46131-1_8
    """

    def mine(self, discrete_sequences: np.ndarray) -> List[np.array]:
        """
        Mine frequent sequential patterns in the given symbolic representation
        of the time series. First, the discrete subsequences are written to a
        temporary file, which serves as input to the SPMF-library. Second, a
        command is constructed and called to use the SPMF-library for mining
        patterns. Finally, the patterns are read from the file to which the
        SPMF-library writes the mined patterns, and the temporary files are
        removed.

        Parameters
        ----------
        discrete_sequences: np.array of shape (n_symbolic_sequences, length_symbolic_sequences)
            The discrete representation of a time series. This representation
            consists of ´n_symbolic_sequences´ subsequences, each one having
            ´length_symbolic_sequences´ symbols. The sequences are provided
            as the rows of the given input matrix.

        Returns
        -------
        patterns: List[np.array]
            The mined, frequent sequential patterns
        """
        # Create an input file and write the discrete subsequences to it
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            input_file_name = tmp_file.name
            tmp_file.write(str.encode(self._encode_input_string(discrete_sequences)))

        # Create an output file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            output_file_name = tmp_file.name

        # Execute the java command to mine the sequences
        command = f"java -jar {pathlib.Path(__file__).parent}/spmf.jar run {self.mining_algorithm()} {input_file_name} {output_file_name} {self.hyperparameters()}"
        output = subprocess.run(command, shell=True, capture_output=True)

        # Read the output file
        with open(output_file_name, 'r') as output_file:
            patterns = self._decode_output_string(output_file.readlines())

        # Raise the exception if something went wrong
        if output.stderr and len(patterns) == 0:
            raise Exception(output.stderr.decode())

        # Clean up the files
        os.remove(input_file_name)
        os.remove(output_file_name)

        # Return the patterns
        return patterns

    @abc.abstractmethod
    def mining_algorithm(self) -> str:
        """
        Return the name of the mining algorithm.

        Returns
        -------
        name: str
            The name of the mining algorithm
        """

    @abc.abstractmethod
    def hyperparameters(self) -> str:
        """
        Return a string-representation of the hyperparameters, which can be used
        for calling the SPMF-library.

        Returns
        -------
        name: str
            The hyperparameters of the pattern miner, as a string
        """

    @abc.abstractmethod
    def _encode_input_string(self, discrete_sequences: np.ndarray) -> str:
        """
        Encode the given discrete subsequences to a string representation,
        which can be interpreted by the algorithm in the SPMF-library.

        Parameters
        ----------
        discrete_sequences: np.array of shape (n_symbolic_sequences, length_symbolic_sequences)
            The discrete representation of a time series, which should be
            encoded to a string.

        Returns
        -------
        encoded_sequences: str
            A string-encoded representation of the given discrete subsequences,
            which can be written to a file and interpreted by the SPMF-library.
        """

    @abc.abstractmethod
    def _decode_output_string(self, output_lines: List[str]) -> List[np.array]:
        """
        Decode the given input strings back to the mined sequential patterns.

        Parameters
        ----------
        output_lines: List[str]
            The lines which have been written to the output by the SPMF-library
            when mining frequent sequential patterns.

        Returns
        -------
        patterns: List[np.array]
            A list consisting of the decoded mined frequent sequential patterns.
        """
