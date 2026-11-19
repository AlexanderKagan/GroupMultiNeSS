from abc import abstractmethod

import numpy as np
from scipy.special import gammaln
from typing import List, Callable, Optional, Union
import matplotlib.pyplot as plt
from functools import partial
from threading import RLock
from copy import copy, deepcopy
from itertools import product
from contextlib import closing

from .utils import (
    soft_thresholding_operator, hard_thresholding_operator,
    frobenius_error, mean_frobenius_error, mark_intervals_on_xaxis,
    if_scalar_or_given_length_array, fill_diagonals, MultipleNetworkTrainTestSplitter,
)
from .base import BaseMultiplexNetworksModel, BaseRefitting, SharedMemoryMatrixFitter
from .MASE import ASE


class BaseMultiNeSS(BaseMultiplexNetworksModel, SharedMemoryMatrixFitter):
    """
    Base class for implementing the MultiNeSS and GroupMultiNeSS models.

    Inherits from
    ----------
    BaseMultiplexNetworksModel
        Provides core multiplex network modeling functionalities.
    SharedMemoryMatrixFitter
        Provides methods for fitting and managing shared memory matrices.
    """

    def get_all_fitted_matrices_by_type(self):
        """
        Retrieve all fitted matrices, separated by shared and individual components.

        Returns
        -------
        list of np.ndarray
            A list containing:
            - The shared latent space matrix.
            - A list of individual latent space matrices.
        """
        return [self.get_shared_latent_space(), self.get_individual_latent_spaces()]

    def get_all_matrix_latent_positions(self, ds: Optional[List[int]] = None, check_if_symmetric=True):
        """
        Compute latent positions for all fitted matrices using adjacency spectral embedding (ASE).

        Parameters
        ----------
        ds : list of int, optional
            Dimensions of embeddings for each matrix. If None, the matrix ranks are used.
        check_if_symmetric : bool, default=True
            Whether to check if matrices are symmetric before embedding.

        Returns
        -------
        list of np.ndarray
            List of latent position matrices for each fitted matrix, one per component.
        """
        matrices = self.get_all_fitted_matrices()
        if ds is None:
            ds = [np.linalg.matrix_rank(matrix) for matrix in matrices]
        return [ASE(matrix, d, check_if_symmetric=check_if_symmetric) if d != 0 else np.empty((self.n_nodes_, 0))
                for matrix, d in zip(matrices, ds, strict=True)]

    def get_node_latent_positions_across_layers(self):
        """
        Compute node latent positions across all layers, concatenating shared and relevant individual embeddings.

        Returns
        -------
        list of np.ndarray
            A list of latent position matrices, one for each layer. Each matrix has shape (n_nodes, total_dim).
        """
        matrix_positions = self.get_all_matrix_latent_positions()
        all_param_indices = np.arange(len(matrix_positions))
        latent_positions = []
        for idx in range(self.n_layers_):
            layer_depend_indices = all_param_indices[self._param_participance_masks[:, idx]]
            layer_pos = np.hstack([matrix_positions[param_idx] for param_idx in layer_depend_indices])
            latent_positions.append(layer_pos)
        return latent_positions

    def get_shared_latent_space(self):
        """
        Retrieve the shared latent space matrix.

        Returns
        -------
        np.ndarray
            The shared latent space matrix estimated across all layers.
        """
        return self.get_all_fitted_matrices()[0]

    def get_individual_latent_spaces(self):
        """
        Retrieve the individual latent space matrices.

        Returns
        -------
        list of np.ndarray
            List of latent space matrices specific to each network layer.
        """
        return self.get_all_fitted_matrices()[1:]

    def _get_param_participance_masks(self):
        """
        Construct boolean masks indicating which parameters (shared or individual)
        contribute to each network layer.

        Returns
        -------
        np.ndarray of bool, shape (n_parameters, n_layers)
            Participation mask where True indicates inclusion of a parameter in a layer.
        """
        return np.stack([np.ones(self.n_layers_, dtype=bool), *np.eye(self.n_layers_, dtype=bool)])

    @staticmethod
    def _get_all_fitted_parameter_names():
        """
        Get the names of all fitted parameter types.

        Returns
        -------
        list of str
            Names of the model components: shared and individual.
        """
        return ["Shared component", "Individual components"]

    def compute_expected_adjacency(self):
        """
        Compute the expected adjacency matrices from the fitted latent spaces.

        Returns
        -------
        list of np.ndarray
            Expected adjacency matrices obtained via the model link function.
        """
        expected = self.link_(self.compute_latent_spaces())
        if not self.loops_allowed:
            fill_diagonals(expected, val=0.)
        return expected

    def compute_latent_spaces(self):
        """
        Compute latent spaces combining shared and individual components,
        optionally zeroing diagonals if loops are not allowed.

        Returns
        -------
        list of np.ndarray
            Latent space matrices adjusted for model constraints.
        """
        ls = self.get_shared_latent_space() + self.get_individual_latent_spaces()
        if not self.loops_allowed:
            fill_diagonals(ls, val=0., inplace=True)
        return ls


class BaseMultiNeSSRefined(BaseMultiNeSS, BaseRefitting):
    """Coordinate descent, initialization, and history shared by both models."""

    def __init__(self, edge_distrib="normal", loops_allowed=True,
                 max_rank=None, init_rank=None, sigmas=None):
        super().__init__(edge_distrib=edge_distrib, loops_allowed=loops_allowed)
        self.max_rank = max_rank
        self.init_rank = init_rank
        self.sigmas = sigmas

    def _initialization_data(self, As):
        # Work in the natural-parameter space for non-identity links.
        if self.edge_distrib == "bernoulli":
            probabilities = np.clip(As, 1e-3, 1. - 1e-3)
            As = np.log(probabilities / (1. - probabilities))
        elif self.edge_distrib == 'poisson':
            # A positive floor makes zero-count initializers finite. The fitted
            # log link and likelihood themselves are not clipped.
            As = np.log(np.maximum(As, .5))
        if not self.loops_allowed:
            As = As.copy()
            fill_diagonals(As, val=0.)
        return As

    def _optimized_params_init(self, As, key=0):
        """Initialize S from the average, then R from layer residuals."""
        As = self._initialization_data(As)
        shared = hard_thresholding_operator(np.nansum(As, axis=0) / self.n_layers_,
                                           max_rank=self.init_rank)
        self._set_matrix(shared, 0)
        for layer, A in enumerate(As):
            residual = np.where(np.isnan(A), 0., A - shared)
            self._set_matrix(hard_thresholding_operator(residual, max_rank=self.init_rank), layer + 1)

    def _update_history(self, nll, key, outer_iter=0,
                        save_nll_history=True, save_latent_history=False):
        history_key = self._key_2_history_key[key]
        if save_nll_history:
            self.nll_history_[outer_iter][history_key].append(nll)
        if save_latent_history:
            self.latent_history_[outer_iter][history_key].append(
                [matrix.copy() for matrix in self.get_all_fitted_matrices_by_type()])

    def _make_key_2_history_key_dict(self):
        self._key_2_history_key = {0: "Shared component"}

    def _make_key_2_param_indices_dict(self):
        # Individual matrices first, shared matrix last.
        self._key_2_param_indices = {0: [*range(1, self.n_fitted_matrices), 0]}

    def _make_key_2_hyperparams_dict(self):
        self._key_2_hyperparams_dict = {}

    def _init_keys(self):
        self._make_key_2_history_key_dict()
        self._make_key_2_param_indices_dict()
        self._make_key_2_hyperparams_dict()

    def _set_hyperparams(self, params_dict, key):
        for name, value in params_dict.items():
            old_value = self._key_2_hyperparams_dict[key][name]
            self._key_2_hyperparams_dict[key][name] = (
                float(value) if np.isscalar(old_value) else np.full_like(old_value, value))

    def _init_history(self, n_outer_iter=1):
        self._shm_lock = RLock()
        self.optim_info = {'n_svd_calls': 0}
        self.nll_history_ = [{name: [] for name in self._key_2_history_key.values()}
                             for _ in range(n_outer_iter)]
        self.latent_history_ = [{name: [] for name in self._key_2_history_key.values()}
                                for _ in range(n_outer_iter)]
        self.convergence_history_ = [{key: [] for key in self._key_2_history_key}
                                    for _ in range(n_outer_iter)]
        self.n_outer_iter_ = 0

    def _validate_hyperparams(self):
        for name in ("init_rank", "max_rank"):
            rank = getattr(self, name)
            if rank is not None and (not isinstance(rank, (int, np.integer)) or
                                     not 0 <= rank <= self.n_nodes_):
                raise ValueError(f"{name} must be an integer between 0 and n_nodes, or None")
        if self.max_rank is None:
            self.max_rank = int(np.sqrt(self.n_nodes_))
        # init_rank only controls initialization; None leaves averages untruncated.
        if self.sigmas is None:
            self.sigmas_ = np.ones((self.n_layers_, self.n_nodes_, self.n_nodes_))
        else:
            sigmas = np.asarray(self.sigmas, dtype=float)
            if sigmas.ndim == 1:
                if sigmas.shape != (self.n_layers_,):
                    raise ValueError("sigmas must have one value per layer")
                sigmas = sigmas[:, None, None]
            self.sigmas_ = np.broadcast_to(sigmas, (self.n_layers_, self.n_nodes_, self.n_nodes_)).copy()
            if not np.all(np.isfinite(self.sigmas_)) or np.any(self.sigmas_ <= 0):
                raise ValueError("sigmas must be finite and positive")

    @staticmethod
    def _validate_penalty(value, name):
        if np.any(~np.isfinite(value)) or np.any(np.asarray(value) < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
        return value

    @abstractmethod
    def _compute_projected_map(self, idx, lr, key) -> Callable:
        raise NotImplementedError

    def _learning_rate(self, lr):
        """Common scale; matrix updates divide it by participation count once."""
        lr = (3. if self.edge_distrib == 'bernoulli' else 1.) if lr is None else lr
        if np.ndim(lr) != 0:
            raise ValueError('lr must be a positive finite scalar')
        lr = float(lr)
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError('lr must be a positive finite scalar')
        return lr

    def _update_matrix(self, idx, lr, sample_grads, key):
        # Both the gradient and proximal threshold use eta = lr / count.
        # Pass the unnormalized common scale to _compute_projected_map.
        mask = self._param_participance_masks[idx]
        matrix_grad = np.nansum(sample_grads[mask], axis=0)
        if not self.loops_allowed:
            np.fill_diagonal(matrix_grad, 0.)
        cur_matrix = self.get_all_fitted_matrices()[idx]
        projected_map = self._compute_projected_map(idx, lr=lr, key=key)
        updated = projected_map(cur_matrix - lr / mask.sum() * matrix_grad)
        self._set_matrix(updated, idx)

    @property
    def n_fitted_matrices(self):
        return 1 + self.n_layers_

    def _pre_fit(self, n_outer_iter=1):
        shape = (self.n_fitted_matrices, self.n_nodes_, self.n_nodes_)
        self._init_param_matrices(np.zeros(shape))
        self._validate_hyperparams()
        self._init_keys()
        self._init_history(n_outer_iter=n_outer_iter)

    def _optimization_step(self, param_indices, key, As, lr, iteration, outer_iter,
                           verbose, tol, save_latent_history, save_nll_history):
        for idx in param_indices:
            self._update_matrix(idx, lr=lr, sample_grads=self._nll_sample_grads(As), key=key)
        obs_mask = self._param_participance_masks[param_indices].any(0)
        nll = self._compute_nll(As, obs_mask=obs_mask)
        with self._shm_lock:
            self.optim_info['n_svd_calls'] += len(param_indices)
            self._update_history(nll, key=key, outer_iter=outer_iter,
                                 save_latent_history=save_latent_history,
                                 save_nll_history=save_nll_history)
        if verbose:
            print(f"{self._key_2_history_key[key]}, outer_iter={outer_iter + 1}, "
                  f"iteration={iteration + 1}, NLL={nll:.6g}")
        return nll

    def _optimize_params(self, key, As, lr, max_iter, tol, patience,
                         verbose=False, verbose_interval=1,
                         save_latent_history=False, save_nll_history=True, outer_iter=0):
        """Stop after `patience` consecutive small changes in S or Q_k."""
        stable_iterations = 0
        for iteration in range(max_iter):
            previous = self.get_all_fitted_matrices()[key].copy()
            self._optimization_step(self._key_2_param_indices[key], key=key, As=As, lr=lr,
                                    iteration=iteration, outer_iter=outer_iter, tol=tol,
                                    verbose=verbose and iteration % verbose_interval == 0,
                                    save_latent_history=save_latent_history,
                                    save_nll_history=save_nll_history)
            change = frobenius_error(previous, self.get_all_fitted_matrices()[key], relative=True)
            self.convergence_history_[outer_iter][key].append(change)
            stable_iterations = stable_iterations + 1 if change <= tol else 0
            if stable_iterations >= patience:
                break
        return iteration + 1

    def _run_optimization(self, As, n_outer_iter=1, **optim_kwargs):
        self.n_outer_iter_ = self._optimize_params(As=As, key=0, **optim_kwargs)

    def _fit(self, As, n_outer_iter=1, refit=True, **optim_kwargs):
        self._pre_fit(n_outer_iter=n_outer_iter)
        self._optimized_params_init(As, key=0)
        self._run_optimization(As, n_outer_iter=n_outer_iter, **optim_kwargs)
        if refit:
            self.refit(As)  # One joint refit of all nonzero parameter matrices.

    @staticmethod
    def _validate_optim_params(max_iter, tol, patience, verbose_interval, n_outer_iter=1):
        for name, value in (("max_iter", max_iter), ("patience", patience),
                            ("verbose_interval", verbose_interval), ("n_outer_iter", n_outer_iter)):
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(tol) or tol < 0:
            raise ValueError("tol must be finite and nonnegative")

    def _nll_sample_grads(self, As):
        """
        Compute sample gradients for the negative log-likelihood objective.

        Parameters
        ----------
        As : np.ndarray
            Observed adjacency matrices.

        Returns
        -------
        np.ndarray
            Gradient tensor for each layer.
        """
        resids = self.compute_expected_adjacency() - As
        if self.edge_distrib == "normal":
            resids /= self.sigmas_ ** 2
        return resids

    def _compute_nll(self, As, obs_mask=None, eps=1e-8):
        """
        Compute negative log-likelihood for observed data under the model.

        Parameters
        ----------
        As : np.ndarray
            Observed adjacency matrices.
        obs_mask : np.ndarray, optional
            Boolean mask for observed layers.
        eps : float, default=1e-8
            Small constant for numerical stability.

        Returns
        -------
        float
            Negative log-likelihood value.
        """
        if obs_mask is None:
            obs_mask = np.ones(self.n_layers_, dtype=bool)
        rows, cols = np.triu_indices(self.n_nodes_, k=0 if self.loops_allowed else 1)
        if self.edge_distrib == 'poisson':
            counts = As[obs_mask][:, rows, cols]
            observed = ~np.isnan(counts)
            theta = self.compute_latent_spaces()[obs_mask][:, rows, cols][observed]
            counts = counts[observed]
            with np.errstate(over='ignore'):
                return np.sum(np.exp(theta) - counts * theta + gammaln(counts + 1.))
        Ps = self.compute_expected_adjacency()
        if self.edge_distrib == "normal":
            full_nll = (As - Ps) ** 2 / (2. * self.sigmas_ ** 2)
        else:
            Ps = np.clip(Ps, eps, 1. - eps)
            full_nll = -As * np.log(Ps) - (1. - As) * np.log(1. - Ps)
        return np.nansum(full_nll[obs_mask][:, rows, cols])

    def make_final_error_report(self, *true_params, Ps=None, relative_errors=True, round_digits: int = 3):
        """
        Compute final error report comparing estimated and true parameters.

        Parameters
        ----------
        *true_params : np.ndarray
            Ground-truth parameter matrices (shared, individual, etc.).
        Ps : np.ndarray, optional
            True expected adjacency matrices.
        relative_errors : bool, default=True
            Whether to compute relative Frobenius errors.
        round_digits : int, default=3
            Number of digits to round results.

        Returns
        -------
        dict
            Mapping of parameter names to error metrics.
        """
        estimate_params = self.get_all_fitted_matrices_by_type()
        param_names = self._get_all_fitted_parameter_names()
        assert len(true_params) == len(estimate_params)
        assert all([param.shape == pred_param.shape for param, pred_param in zip(true_params, estimate_params)])
        if Ps is not None:
            true_params = [*true_params, Ps]
            estimate_params.append(self.compute_expected_adjacency())
            param_names.append("Ps")
        return {param_name: round(mean_frobenius_error(true_param, pred_param,
                                                       relative=relative_errors, include_offdiag=self.loops_allowed),
                                  round_digits)
                for param_name, true_param, pred_param in zip(param_names, true_params, estimate_params)}

    def get_error_history(self, *true_params, relative_errors=True):
        """Return error curves and inner-loop boundaries, indexed by integer keys."""
        if not any(values for history in self.latent_history_ for values in history.values()):
            raise ValueError("Fit with save_latent_history=True to access latent history")
        if len(true_params) != len(self._get_all_fitted_parameter_names()):
            raise ValueError("Supply one true array per component type")
        result = {}
        for key, name in self._key_2_history_key.items():
            series, ends, count = {}, [], 0
            for history in self.latent_history_:
                snapshots = history[name]
                for estimates in snapshots:
                    for label, truth, estimate in self._error_components(key, true_params, estimates):
                        error = mean_frobenius_error(np.asarray(truth), np.asarray(estimate),
                                                    relative=relative_errors,
                                                    include_offdiag=self.loops_allowed)
                        series.setdefault(label, []).append(error)
                if snapshots:
                    count += len(snapshots)
                    ends.append(count)
            result[key] = {"series": {label: np.asarray(vals) for label, vals in series.items()},
                           "outer_iter_ends": ends}
        return result

    def _error_components(self, key, true_params, estimates):
        return zip((r"$S$", r"$R_\ell$"), true_params, estimates)

    def _plot_history(self, histories, ylabel, n_cols, fontsize, figsize, shared_title=None):
        keys = sorted(histories)
        n_cols = min(n_cols, len(keys))
        fig, axs = plt.subplots(ncols=n_cols, nrows=int(np.ceil(len(keys) / n_cols)), figsize=figsize)
        axs = np.asarray(axs).reshape(-1)
        for ax, key in zip(axs, keys):
            history = histories[key]
            for label, values in history['series'].items():
                ax.plot(np.arange(1, len(values) + 1), values, label=label)
            if key == 0:
                ax.set_xlabel('outer_iter', fontsize=fontsize)
            else:
                mark_intervals_on_xaxis(ax, history['outer_iter_ends'], fontsize=fontsize)
            title = shared_title if key == 0 and shared_title is not None else self._key_2_history_key[key]
            ax.set_title(title, fontsize=fontsize)
            ax.set_ylabel(ylabel, fontsize=fontsize)
            ax.tick_params(axis='both', labelsize=fontsize)
            if len(history['series']) > 1:
                ax.legend(fontsize=fontsize)
        for ax in axs[len(keys):]:
            fig.delaxes(ax)
        fig.tight_layout(h_pad=5)
        return fig, axs

    def plot_error_history(self, *true_params, relative_errors=True, n_cols=4,
                           fontsize=14, figsize=(12, 20), print_final_errors=True):
        histories = self.get_error_history(*true_params, relative_errors=relative_errors)
        fig, axs = self._plot_history(histories, 'Error', n_cols, fontsize, figsize)
        if print_final_errors:
            print('Final errors:', {self._key_2_history_key[key]:
                  {name: round(float(values[-1]), 3) for name, values in hist['series'].items()}
                  for key, hist in histories.items()})
        plt.show()
        return fig, axs

    def plot_nll_history(self, n_cols=4, fontsize=14, figsize=(12, 20)):
        """Plot unpenalized joint/group NLL over outer/inner iterations.

        Nuclear-norm shrinkage can increase NLL while reducing the penalized
        objective; these curves alone are not a monotonicity diagnostic.
        """
        histories = {}
        for key, name in self._key_2_history_key.items():
            values, ends = [], []
            for history in self.nll_history_:
                if history[name]:
                    values.extend(history[name])
                    ends.append(len(values))
            histories[key] = {'series': {'NLL': np.asarray(values)}, 'outer_iter_ends': ends}
        fig, axs = self._plot_history(histories, 'Negative log-likelihood', n_cols, fontsize, figsize,
                                     shared_title='Outer-loop convergence')
        plt.show()
        return fig, axs


class BaseMultiNeSSRefinedCV(BaseMultiNeSSRefined):
    """Edge CV with independent training initialization and candidate state.

    ``shared`` searches rho_0 and links all other penalties by their oracle rates.
    ``full`` searches each rho_k (with rho_kl = rho_k / sqrt(m_k)), then rho_0,
    using the first outer iteration only. Selected penalties are reused thereafter.
    cv_folds is the number of repeated edge holdouts of size test_prop; None uses
    one holdout. No held-out edge is used to initialize a training fit.
    """

    def __init__(self, param_grid=None, cv_folds=3, test_prop=0.1,
                 fix_layer_split=False, use_1se_rule=False, random_seed=None,
                 cv_type='shared', **kwargs):
        super().__init__(**kwargs)
        if cv_type not in ('shared', 'full'):
            raise ValueError("cv_type must be 'shared' or 'full'")
        if cv_folds is not None and (not isinstance(cv_folds, (int, np.integer)) or cv_folds < 1):
            raise ValueError("cv_folds must be a positive integer or None")
        if not 0 < test_prop < 1:
            raise ValueError("test_prop must lie strictly between 0 and 1")
        self.param_grid = param_grid
        self.cv_type = cv_type
        self.cv_folds = cv_folds
        self.test_prop = test_prop
        self.fix_layer_split = fix_layer_split
        self.use_1se_rule = use_1se_rule
        self.random_seed = random_seed

    def _init_keys(self):
        super()._init_keys()
        self._make_key_2_hyperparams_grid_dict()

    def _make_key_2_hyperparams_grid_dict(self):
        self._key_2_hyperparams_grid = {0: {'rho_0': self._penalty_grid('rho_0',
                                      np.sqrt(self.n_nodes_ * self.n_layers_))}}

    def _penalty_grid(self, name, scale, group=None):
        grid = None if self.param_grid is None else self.param_grid.get(name)
        if isinstance(grid, dict):
            if group is None:
                raise ValueError(f"{name} requires a single grid")
            grid = grid.get(group)
        if grid is None:
            grid = scale * np.array([0.03, 0.1, 0.3, 1., 3., 10.])
        grid = np.asarray(grid, dtype=float)
        if grid.ndim != 1 or not len(grid) or len(np.unique(grid)) != len(grid):
            raise ValueError(f"{name} grid must be a nonempty sequence of distinct scalars")
        self._validate_penalty(grid, name)
        return grid

    def _init_history(self, n_outer_iter=1):
        super()._init_history(n_outer_iter=n_outer_iter)
        self._cv_results = {key: {params: [] for params in product(*grid.values())}
                            for key, grid in self._key_2_hyperparams_grid.items()}
        self._best_params_dict = {}

    def _set_best_hyperparams(self, key):
        results = self._cv_results[key]
        params = list(results)
        scores = np.asarray(list(results.values()))
        means = scores.mean(axis=1)
        best_idx = int(np.argmin(means))
        if self.use_1se_rule and scores.shape[1] > 1:
            upper = means[best_idx] + scores[best_idx].std(ddof=1) / np.sqrt(scores.shape[1])
            best_idx = max((i for i, mean in enumerate(means) if mean <= upper), key=lambda i: params[i])
        selected = dict(zip(self._key_2_hyperparams_grid[key], params[best_idx]))
        self._best_params_dict[key] = selected
        self._set_cv_penalties(selected, key)

    def _set_cv_penalties(self, params, key):
        self._set_hyperparams(params, key)
        if self.cv_type == 'shared':
            self._key_2_hyperparams_dict[0]['rho_kl'] = np.full(
                self.n_layers_, params['rho_0'] / np.sqrt(self.n_layers_))

    def _cv_search(self, As, n_outer_iter, refit, optim_kwargs):
        self._cv_shared(As, n_outer_iter, refit, optim_kwargs)

    def _cv_splits(self, As):
        rng = np.random.default_rng(self.random_seed)
        for _ in range(self.cv_folds or 1):
            splitter = MultipleNetworkTrainTestSplitter(
                loops_allowed=self.loops_allowed, fix_layer_split=self.fix_layer_split,
                random_seed=int(rng.integers(0, 2**32 - 1)))
            train, test = splitter.train_test_split(As, test_prop=self.test_prop)
            indices = np.triu_indices(self.n_nodes_, k=0 if self.loops_allowed else 1)
            for key, param_indices in self._key_2_param_indices.items():
                mask = self._param_participance_masks[param_indices].any(0)
                for subset in (train, test):
                    if not np.any(np.isfinite(subset[mask][:, indices[0], indices[1]])):
                        raise ValueError(f"Empty training or test edge set for key {key}; "
                                         "adjust test_prop or random_seed")
            yield train, test

    def _new_cv_model(self, train, n_outer_iter):
        candidate = copy(self)
        # A shallow model copy must never reuse or close the parent's allocation.
        candidate._param_matrices_shm = None
        candidate.param_matrices = None
        candidate._pre_fit(n_outer_iter=n_outer_iter)
        candidate._optimized_params_init(train, key=0)
        return candidate

    def _cv_shared(self, As, n_outer_iter, refit, optim_kwargs):
        for train, test in self._cv_splits(As):
            with closing(self._new_cv_model(train, n_outer_iter)) as candidate:
                initial = candidate.get_all_fitted_matrices().copy()
                for params in self._cv_results[0]:
                    candidate._set_matrices(initial)
                    candidate._set_cv_penalties(dict(zip(self._key_2_hyperparams_grid[0], params)), 0)
                    candidate._run_optimization(train, n_outer_iter=n_outer_iter, **optim_kwargs)
                    if refit:
                        candidate.refit(train)
                    self._cv_results[0][params].append(candidate._compute_nll(test))
        self._set_best_hyperparams(0)

    def _fit(self, As, n_outer_iter=1, refit=True, **optim_kwargs):
        self._pre_fit(n_outer_iter=n_outer_iter)
        cv_kwargs = {**optim_kwargs, 'save_latent_history': False, 'save_nll_history': False,
                     'verbose': False}
        self._cv_search(As, n_outer_iter, refit, cv_kwargs)
        self._optimized_params_init(As, key=0)
        self._run_optimization(As, n_outer_iter=n_outer_iter, **optim_kwargs)
        if refit:
            self.refit(As)

    def get_cv_results(self):
        return deepcopy(self._cv_results)

    def get_best_params_dict(self):
        return deepcopy(self._best_params_dict)

    def plot_cv_results(self, log_params=True, fontsize=12):
        """Plot held-out NLL against actual rho values (log10 by default)."""
        results = self.get_cv_results()
        fig, axs = plt.subplots(1, len(results), figsize=(6 * len(results), 5), squeeze=False)
        for ax, (key, values) in zip(axs[0], sorted(results.items())):
            grid = np.asarray(list(values)).ravel()
            if log_params and np.any(grid <= 0):
                raise ValueError("Logarithmic CV plots require positive penalties; use log_params=False")
            grid = np.log10(grid) if log_params else grid
            scores = np.asarray(list(values.values()))
            for fold in range(scores.shape[1]):
                ax.plot(grid, scores[:, fold], marker='o', label=f'Fold={fold + 1}')
            name = rf'$\rho_{{{key}}}$'
            ax.set_xlabel(r'$\log_{10}$' + f'({name})' if log_params else name, fontsize=fontsize)
            ax.set_ylabel('NLL', fontsize=fontsize)
            if key != 0:
                ax.set_title(self._key_2_history_key[key], fontsize=fontsize)
            ax.legend()
        fig.tight_layout()
        plt.show()
        return fig, axs[0]


class MultiNeSS(BaseMultiNeSSRefined, BaseRefitting):
    """MultiNeSS with direct nuclear-norm penalties rho_0 and rho_kl.

    None penalties become 2*sqrt(M*n) for S and 2*sqrt(n) for each R_l.
    rho_kl may be a scalar or a sequence in input-layer order. init_rank=None
    leaves the averaging initializer untruncated; max_rank limits fitted ranks.
    """

    def __init__(self, rho_0=None, rho_kl=None, max_rank=None, init_rank=None,
                 edge_distrib='normal', sigmas=None, loops_allowed=True, refit_threshold=1e-8):
        BaseMultiNeSSRefined.__init__(self, edge_distrib=edge_distrib, max_rank=max_rank,
                                      init_rank=init_rank, sigmas=sigmas, loops_allowed=loops_allowed)
        BaseRefitting.__init__(self, edge_distrib=edge_distrib, max_rank=max_rank,
                               loops_allowed=loops_allowed, refit_threshold=refit_threshold)
        self.rho_0 = rho_0
        self.rho_kl = rho_kl

    def _compute_penalty_coef(self, idx, key):
        params = self._key_2_hyperparams_dict[key]
        return params['rho_0'] if idx == 0 else params['rho_kl'][idx - 1]

    def _compute_projected_map(self, idx, lr, key):
        penalty = self._compute_penalty_coef(idx, key)
        mask = self._param_participance_masks[idx]
        return partial(soft_thresholding_operator, threshold=penalty * lr / mask.sum(), max_rank=self.max_rank)

    def _make_key_2_hyperparams_dict(self):
        rho_0 = 2. * np.sqrt(self.n_layers_ * self.n_nodes_) if self.rho_0 is None else self.rho_0
        if not np.isscalar(rho_0):
            raise ValueError('rho_0 must be a scalar')
        rho_kl = if_scalar_or_given_length_array(
            2. * np.sqrt(self.n_nodes_) if self.rho_kl is None else self.rho_kl,
            length=self.n_layers_, name='rho_kl')
        self._key_2_hyperparams_dict = {0: {
            'rho_0': self._validate_penalty(rho_0, 'rho_0'),
            'rho_kl': self._validate_penalty(rho_kl, 'rho_kl')}}

    def fit(self, As, lr=None, max_iter=100, tol=1e-3, patience=10, refit=True,
            verbose=True, verbose_interval=1, save_latent_history=False, save_nll_history=True):
        """Fit R then S until S's successive RFE stays below tol for patience steps.

        lr scales the participation-normalized steps: 1 for Gaussian/Poisson
        and 3 for Bernoulli. Steps are fixed, without backtracking. All matrices
        are jointly refitted at the end.
        """
        self._validate_optim_params(max_iter, tol, patience, verbose_interval)
        lr = self._learning_rate(lr)
        As = self._validate_input(As)
        self._fit(As, lr=lr, max_iter=max_iter, tol=tol, patience=patience, refit=refit,
                  verbose=verbose, verbose_interval=verbose_interval,
                  save_latent_history=save_latent_history, save_nll_history=save_nll_history)
        return self


class MultiNeSSCV(MultiNeSS, BaseMultiNeSSRefinedCV):
    """CV for MultiNeSS; param_grid contains rho_0 candidates.

    shared links rho_kl to rho_0/sqrt(M); full tunes rho_0 with fixed rho_kl
    (there are no group penalties to search in MultiNeSS).
    """

    def __init__(self, param_grid=None, cv_folds=3, test_prop=0.1,
                 rho_0=None, rho_kl=None, edge_distrib='normal', max_rank=None,
                 init_rank=None, sigmas=None, loops_allowed=True, refit_threshold=1e-8,
                 fix_layer_split=False, use_1se_rule=False, random_seed=None, cv_type='shared'):
        BaseMultiNeSSRefinedCV.__init__(self, param_grid=param_grid, cv_folds=cv_folds,
                                        test_prop=test_prop, fix_layer_split=fix_layer_split,
                                        use_1se_rule=use_1se_rule, random_seed=random_seed,
                                        cv_type=cv_type, loops_allowed=loops_allowed)
        MultiNeSS.__init__(self, rho_0=rho_0, rho_kl=rho_kl, edge_distrib=edge_distrib,
                           max_rank=max_rank, init_rank=init_rank, sigmas=sigmas,
                           loops_allowed=loops_allowed, refit_threshold=refit_threshold)

    def _make_key_2_hyperparams_grid_dict(self):
        if self.param_grid is not None and set(self.param_grid) - {'rho_0'}:
            raise ValueError('MultiNeSSCV param_grid only accepts rho_0')
        BaseMultiNeSSRefinedCV._make_key_2_hyperparams_grid_dict(self)


class BaseOracleMultiNeSS(BaseMultiNeSSRefined):
    """
    Oracle MultiNeSS with rank thresholding.

    Parameters
    ----------
    Inherits all parameters from BaseMultiNeSSRefined.
    """

    @abstractmethod
    def _compute_threshold_rank(self, idx: int, key: int) -> int:
        """
        Compute the threshold rank for hard-thresholding operator.

        Parameters
        ----------
        idx : int
            Parameter matrix index.
        key : int
            Network-layer key.

        Returns
        -------
        int
            Threshold rank.
        """
        pass

    def _compute_projected_map(self, idx: int, lr, key: int):
        """
        Generate a projected update function (hard-thresholding) for a parameter.

        Parameters
        ----------
        idx : int
            Index of the parameter matrix.
        lr : float
            Learning rate.
        key : int
            Network-layer key.

        Returns
        -------
        callable
            Function applying hard-thresholding with computed max rank.
        """
        rank = self._compute_threshold_rank(idx=idx, key=key)
        return partial(hard_thresholding_operator, max_rank=rank)

    def fit(self, As: Union[List[np.ndarray], np.ndarray],
            lr: float = 1.,
            max_iter: int = 100, tol=1e-3, patience=10,
            verbose=True, verbose_interval=1,
            save_latent_history=False, save_nll_history=True, n_outer_iter=10):
        """
        Fit the oracle MultiNeSS model.

        Parameters
        ----------
        As : list of np.ndarray
            Observed adjacency matrices.
        lr : float, default=1.
            Learning rate for optimization.
        max_iter : int, default=100
            Maximum number of iterations.
        tol : float, default=1e-5
            Convergence tolerance.
        patience : int, default=10
            Patience for early stopping.
        verbose : bool, default=True
            Print progress messages.
        verbose_interval : int, default=1
            Verbose print interval.
        save_latent_history : bool, default=False
            Whether to store latent matrices at each iteration.
        save_nll_history : bool, default=True
            Whether to store NLL at each iteration.

        Returns
        -------
        self
            Fitted oracle MultiNeSS model.
        """
        self._validate_optim_params(max_iter, tol, patience, verbose_interval, n_outer_iter)
        lr = self._learning_rate(lr)
        As = self._validate_input(As)
        self._fit(As, lr=lr, n_outer_iter=n_outer_iter,
                  max_iter=max_iter, tol=tol, patience=patience,
                  verbose=verbose, verbose_interval=verbose_interval,
                  refit=False,
                  save_latent_history=save_latent_history, save_nll_history=save_nll_history)
        return self


class OracleMultiNeSS(BaseOracleMultiNeSS):
    """
    Oracle MultiNeSS model with fixed shared and individual ranks.

    Inherits from `BaseOracleMultiNeSS` and sets the threshold ranks based on
    user-specified shared and individual layer dimensions.

    Parameters
    ----------
    d_shared : int
        Rank of the shared component across all layers.
    d_individs : int or list of int
        Ranks of the individual components for each layer.
    edge_distrib : str, default='normal'
        Edge distribution ('normal', 'bernoulli', or 'poisson').
    loops_allowed : bool, default=True
        Whether self-loops are allowed in the network.
    """

    def __init__(self, d_shared: int, d_individs: Union[int, List[int]],
                 edge_distrib: str = "normal", loops_allowed: bool = True):
        """
        Initialize the OracleMultiNeSS model.

        Parameters
        ----------
        d_shared : int
            Rank of the shared component.
        d_individs : int or list of int
            Rank(s) of individual components for each layer.
        edge_distrib : str, default='normal'
            Edge distribution.
        loops_allowed : bool, default=True
            Whether self-loops are allowed.
        """
        BaseMultiNeSSRefined.__init__(self, edge_distrib=edge_distrib, loops_allowed=loops_allowed)
        self.d_shared = d_shared
        self.d_individs = d_individs

    def _make_key_2_hyperparams_dict(self):
        """
        Initialize the mapping from network-layer key to hyperparameters.
        """
        d_shared = int(if_scalar_or_given_length_array(self.d_shared, length=1, name="d_shared")[0])
        d_individs = if_scalar_or_given_length_array(self.d_individs, length=self.n_layers_, name="d_individs")
        self._key_2_hyperparams_dict = {0: {"d_shared": d_shared, "d_individs": d_individs}}

    def _compute_threshold_rank(self, idx: int, key: int) -> int:
        """
        Compute the threshold rank for the hard-thresholding operator.

        Parameters
        ----------
        idx : int
            Index of the parameter matrix (0 for shared component, >0 for individual layers).
        key : int
            Network-layer key.

        Returns
        -------
        int
            Rank threshold for hard-thresholding.
        """
        hyperparam_dict = self._key_2_hyperparams_dict[key]
        d_shared, d_individs = hyperparam_dict["d_shared"], hyperparam_dict["d_individs"]
        return int(d_shared if idx == 0 else d_individs[idx - 1])
