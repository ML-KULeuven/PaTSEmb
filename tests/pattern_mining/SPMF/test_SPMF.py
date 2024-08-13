
import pytest
import numpy as np
from patsemb.pattern_mining.SPMF.QCSP import QCSP


class TestSPMF:

    @pytest.mark.parametrize('seed', range(5))
    def test_raise_error_subprocess(self, seed):
        np.random.seed(seed)

        pattern_miner = QCSP()
        symbolic = np.random.randint(5, size=(100, 5))

        try:
            pattern_miner.mine(symbolic)
        except Exception:
            pytest.fail('An exception has been raised while this should not happen!')

        with pytest.raises(Exception):
            pattern_miner.mine(symbolic.astype(float))
