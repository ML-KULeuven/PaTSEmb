# PaTSEmb

[![PyPi Version](https://img.shields.io/pypi/v/patsemb.svg)](https://pypi.org/project/patsemb/)
[![Downloads](https://static.pepy.tech/badge/patsemb)](https://pepy.tech/project/patsemb)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/patsemb)](https://pypi.python.org/pypi/patsemb/)
[![PyPI license](https://img.shields.io/pypi/l/patsemb.svg)](https://pypi.python.org/pypi/patsemb/)

Welcome to ``PaTSEmb``, a fast and extendable Python package for creating a pattern-based embedding
of the time series. This is an embedding of the time series which contains information
about the typical shapes are occurring at which locations in the time series. 
Below, we give a small example of how to do this, but be sure to check out the 
[documentation](https://PaTSEmb.readthedocs.io/)!

## Installation

You can install ``PaTSEmb`` using the following command:
```
pip install patsemb
```
If you want to mine frequent, sequential patterns, Java 1.7 or higher should also be 
available on your machine. More information about installing ``PaTSEmb`` can be found 
in the [documentation](https://PaTSEmb.readthedocs.io/en/stable/installation.html).

## Example 

The code snippet below shows how to create the pattern-based embedding of a 
time series. Be sure to check out the [example notebook](notebooks/examples) 
for more examples!

```python
from patsemb.discretization import SAXDiscretizer
from patsemb.pattern_mining import QCSP
from patsemb.pattern_based_embedding import PatternBasedEmbedder

# Specify a discretizer and pattern miner, or use the default values
pattern_based_embedder = PatternBasedEmbedder(
    discretizer=SAXDiscretizer(alphabet_size=8, word_size=5),
    pattern_miner=QCSP(minimum_support=3, top_k_patterns=20)
)

# Create the pattern-based embedding
time_series = ...  # Load here your time series as a numpy array
embedding = pattern_based_embedder.fit_transform(time_series)
```

## Contact

Feel free to email to [louis.carpentier@kuleuven.be](mailto:louis.carpentier@kuleuven.be) if 
there are any questions, remarks, ideas, ...

## Acknowledgments 

If you use ``PaTSEmb`` in your research or project, please add the following citation:

```bibtex
@inproceedings{carpentier2024pattern,
    title={Pattern-based Time Series Semantic Segmentation with Gradual State Transitions},
    author={Carpentier, Louis and Feremans, Len and Meert, Wannes and Verbeke, Mathias},
    booktitle={Proceedings of the 2024 SIAM International Conference on Data Mining (SDM)},
    pages={316--324},
    year={2024},
    month={April},
    organization={SIAM},
    doi={10.1137/1.9781611978032.36}
}
```
> L. Carpentier, L. Feremans, W. Meert, and M. Verbeke. 
> "Pattern-based time series semantic segmentation with gradual state transitions". 
> In Proceedings of the 2024 SIAM International Conference on Data Mining (SDM), 
> pages 316–324. SIAM, april 2024. doi: [10.1137/1.9781611978032.36](https://doi.org/10.1137/1.9781611978032.36).

## License

    Copyright (c) 2024 KU Leuven, DTAI Research Group

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.