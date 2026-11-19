# GroupMultiNeSS

GroupMultiNeSS is a package for statistical modeling of multilayer networks. It implements multiple approaches allowing to extract shared, group, and individual latent structures from a collection of networks on a shared set of nodes. Specifically, it contains the implementation of fitting and sampling procedures for the following models:
- GroupMultiNeSS [[Kagan et al. (2025)](https://arxiv.org/abs/2511.11086)] - likelihood based approach with nuclear norm penalization, accounts for the additional group latent structure
- MultiNeSS [[MacDonald et al. (2021)](https://arxiv.org/abs/2012.14409)] - likelihood based approach with nuclear norm penalization
- MultiNeSS [[Tian et al. (2024)](https://arxiv.org/abs/2412.02151)] - likelihood based approach with pre-estimation of latent ranks via Shared Space Hunting algorithm
- COSIE [[Arroyo et al. (2020)](https://arxiv.org/abs/1906.10026)] - spectral-based Multiple Adjacency Spectral Embedding algorithm

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
n, M, K = 200, 16, 4
group_props = np.ones(K) / K  # make ballanced groups
group_indices = make_group_indices(group_props, M)

lpg = GroupLatentPositionGenerator(n_nodes=n, n_layers=M, group_indices=group_indices)
lpg.generate(random_seed=1)

As, Ps_true, S_true, Qs_true, Rs_true = lpg.As, lpg.Ps, lpg.S, lpg.Qs, lpg.Rs

# Fit GroupMultiNeSS model and compute the relative errors with ground-truth
group_multiness = GroupMultiNeSS(group_indices, n_jobs=K)
group_multiness.fit(As, lr=0.8)
print(group_multiness.make_final_error_report(S_true, Qs_true, Rs_true, Ps=Ps_true))

```

## Fitting and cross-validation

The fit follows Algorithm 1 of the updated manuscript: each group alternates
individual updates and a group update, then the shared component is updated once.
The group and outer loops stop after `patience=10` consecutive relative Frobenius
changes below `tol=1e-3` in `Q_k` and `S`, respectively. `max_iter` limits each
group descent and `n_outer_iter=10` (formerly `n_epochs`) limits the outer loop.
Each fit call starts from fresh data-based initialization and resets its history.
The default learning-rate multiplier is 1 for Gaussian/Poisson and 3 for Bernoulli edges;
`lr` is one common scalar multiplier: `eta_S = lr / M`, `eta_Q = lr / m_k`,
and `eta_R = lr`. Each update uses `eta` for the gradient and `rho * eta`
for the threshold, so participation counts are applied exactly once.
All distributions use fixed steps without backtracking. For Poisson edges,
choose a smaller `lr` when expected counts are large: curvature grows with the mean.

The old stage-specific `lmbda`/`alpha` penalties are replaced by direct penalties:

| Argument | Component | Default when `None` |
| --- | --- | --- |
| `rho_0` | Shared `S` | `2 * sqrt(M * n)` |
| `rho_k` | Group `Q_k` | `2 * sqrt(m_k * n)` |
| `rho_kl` | Individual `R_kl` | `2 * sqrt(n)` |

`rho_k` accepts a scalar or one value per group; `rho_kl` accepts a scalar or one
value per input layer. MultiNeSS uses `rho_0` and `rho_kl`. `init_rank` controls
initialization truncation: `None` leaves the averages untruncated, while an integer
uses that rank for `S` and individual residuals and twice that rank for group
averages before subtracting `S`. `max_rank` continues to cap fitted ranks.
With `refit=True`, all nonzero matrices are jointly refitted after optimization.

```python
from GroupMultiNeSS.group_multiness import GroupMultiNeSSCV

model = GroupMultiNeSSCV(group_indices, cv_type="shared", random_seed=1)
model.fit(As, max_iter=100, n_outer_iter=10)
print(model.get_best_params_dict())
```

`cv_type="shared"` searches only `rho_0`, with the other penalties linked as
`rho_k = rho_0 * sqrt(m_k / M)` and `rho_kl = rho_0 / sqrt(M)`. This implements
the manuscript's single common tuning constant. `cv_type="full"` first searches
each `rho_k`, linking `rho_kl = rho_k / sqrt(m_k)`, and then searches `rho_0` using
the same fold-specific training and test edges. This selection uses only the
first outer iteration; the selected penalties are held fixed in the final fit.
For MultiNeSS, `full` searches `rho_0` with fixed individual penalties.

Default candidate grids are `[0.03, 0.1, 0.3, 1, 3, 10]` times the corresponding
square-root rate. Custom grids contain actual penalty values, for example
`param_grid={"rho_0": [5, 10, 20], "rho_k": [3, 6, 12]}` for full CV. A `rho_k`
grid may also be a dictionary of grids indexed by zero-based group labels.
Shared CV accepts only a `rho_0` grid. CV-selected linked penalties replace any
constructor values for those penalties.
CV plots show actual penalties (or their base-10 logarithms by default), not
the common multiplier: `rho_0 = c_rho * sqrt(M * n)`.

`cv_folds=3` performs three repeated edge holdouts, each with `test_prop=0.1`;
`cv_folds=None` uses one holdout. Symmetric edge pairs stay together. Each
candidate starts from training-only initialization, and fold scores and latent
states do not enter the final fit history. When refitting is enabled, completed
CV candidates also receive one joint refit using their training edges.

Optimization, history, and CV result keys are integers: `0` for `S`, and `1..K`
for group descents (group labels themselves remain `0..K-1`). MultiNeSS has only
key `0`. `plot_nll_history()` plots joint NLL over outer iterations for `S` and
concatenated inner descents for each group, titled `Outer-loop convergence` and
`GroupOptimize(k)`, with separated underbraces and 45-degree labels `out_it=...`.
These curves show unpenalized NLL: shrinkage can increase it while decreasing
the penalized objective. Group curves also span updates to `S` between descents.
`plot_error_history()` uses the same layout when fitting with
`save_latent_history=True`; `get_error_history()` includes `outer_iter_ends`.

## Edge distributions and refitting

Both models, their CV versions, and the data generators support:

| `edge_distrib` | Expected edge value | Joint refit solver |
| --- | --- | --- |
| `"normal"` | `S + Q_k + R_kl` | `LinearRegression` (weighted by `1 / sigmas**2` when supplied) |
| `"bernoulli"` | `sigmoid(S + Q_k + R_kl)` | `LogisticRegression` with L-BFGS |
| `"poisson"` | `exp(S + Q_k + R_kl)` | `PoissonRegressor` with L-BFGS |

MultiNeSS omits `Q_k`. Every refit uses **no intercept and no regularization**:
it estimates the eigenvalues of all retained component directions jointly.
These solvers replace repeated GLM IRLS solves in the usual joint-refit path.
Explicit partial refits still account for the fixed matrices through an offset:
Gaussian refits subtract it from the response, while logistic/Poisson partial
refits use statsmodels GLM. Single-class Bernoulli responses also use GLM because
scikit-learn's classifier requires two classes.

```python
model = GroupMultiNeSSCV(
    group_indices, edge_distrib="poisson", cv_type="shared", random_seed=1
)
model.fit(count_adjacencies)
expected_counts = model.compute_expected_adjacency()
```

Poisson inputs must contain nonnegative integer counts; `NaN` marks missing
edges. Initialization takes `log(max(count, 0.5))` so zeros remain finite; the
likelihood and fitted log link are not clipped. Poisson CV uses the Poisson
negative log-likelihood, including the log-factorial term. With
`loops_allowed=False`, predicted and generated adjacency diagonals are zero.

Run the algorithm regression checks with `python -m unittest discover -s tests -v`.


## License

MIT License

Copyright (c) 2025 Alexander Kagan

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
