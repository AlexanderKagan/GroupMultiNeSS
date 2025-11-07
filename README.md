# GroupMultiNeSS

GroupMultiNeSS is a package for statistical modeling of multilayer networks. It implements multiple approaches allowing to extract shared, group, and individual latent structures from a collection of networks on a shared set of nodes. Specifically, it contains fitting procedures for the GroupMultiNeSS, MultiNeSS ([MacDonald et al. (2021)](https://arxiv.org/abs/2012.14409) and [Tian et al. (2024)](https://arxiv.org/abs/2412.02151)), and COSIE [Arroyo et al.] (https://arxiv.org/abs/1906.10026) models.


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install GroupMultiNeSS.

```bash
pip install GroupMultiNeSS
```

## Usage

```python
# Imports
import numpy as np
from GroupMultiNeSS.group_multiness import GroupMultiNeSS
from GroupMultiNeSS.data_generation import GroupLatentPositionGenerator
from GroupMultiNeSS.utils import make_group_indices


# Sample true latent positions and group indices
n = 200
M = 16
K = 4

group_props = np.ones(K) / K  # make ballanced groups
group_indices = make_group_indices(group_props, M)

lpg = GroupLatentPositionGenerator(n_nodes=n, n_layers=M, group_indices=group_indices, edge_distrib="normal")
lpg.generate(random_seed=1)
As, Ps_true, S_true, Qs_true, Rs_true = lpg.As, lpg.Ps, lpg.S, lpg.Qs, lpg.Rs

# Fit GroupMultiNeSS model and compute the relative errors with ground-truth
group_multiness = GroupMultiNeSS(group_indices, n_jobs=K)
group_multiness.fit(As, lr=0.8)
print(group_multiness.make_final_error_report(S_true, Qs_true, Rs_true, Ps=Ps_true))
{'Shared component': 0.026, 'Group components': 0.051, 'Individual components': 0.122, 'Ps': 0.077} 
```


## License

MIT License

Copyright (c) 2024 Alexander Kagan

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
