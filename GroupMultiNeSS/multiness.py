import numpy as np
from typing import List, Dict, Callable, Union, Tuple
import matplotlib.pyplot as plt
from functools import partial
import seaborn as sns
from multiprocessing import Manager
from copy import deepcopy
from itertools import product
from utils import MultipleNetworkTrainTestSplitter
from more_itertools import zip_equal

from .utils import soft_thresholding_operator, hard_thresholding_operator, make_error_report, \
    mean_frobenius_error, EarlyStopper, if_scalar_or_given_length_array, fill_diagonals
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

    def get_all_matrix_latent_positions(self, ds: List[int] = None, check_if_symmetric=True):
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
                for matrix, d in zip_equal(matrices, ds)]

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
        return self.link_(self.compute_latent_spaces())

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
    """
    Refined version of the BaseMultiNeSS model with optimization, refitting, and
    training history management. Implements parameter optimization loops,
    hyperparameter validation, and history tracking during training.

    Inherits from
    ----------
    BaseMultiNeSS
        Provides multiplex latent space modeling functionality.
    BaseRefitting
        Adds post-optimization refitting procedures.
    """

    def __init__(self, edge_distrib: str = "normal", loops_allowed: bool = True,
                 max_rank: int = None, init_rank: int = None,
                 sigmas: Union[float, List, np.ndarray] = None):
        """
        Initialize a refined MultiNeSS model with specified configuration.

        Parameters
        ----------
        edge_distrib : {'normal', 'bernoulli'}, default='normal'
            Distributional assumption for edge weights.
        loops_allowed : bool, default=True
            Whether self-loops are allowed in adjacency matrices.
        max_rank : int, optional
            Maximum rank allowed during optimization.
        init_rank : int, optional
            Initial rank for spectral thresholding.
        sigmas : float or array-like, optional
            Edge variance(s) or standard deviations for Gaussian edges.
        """
        super().__init__(edge_distrib=edge_distrib, loops_allowed=loops_allowed)
        self.max_rank = max_rank
        self.init_rank = init_rank
        self.sigmas = sigmas

    def _optimized_params_init(self, As, key: Tuple[int, int]) -> None:
        """
        Initialize model parameters before optimization via rank-thresholded averaging.

        Parameters
        ----------
        As : np.ndarray
            Input observed adjacency tensors of shape (n_layers, n_nodes, n_nodes).
        key : tuple of int
            Key identifying optimization stage.
        """
        param_indices = self._key_2_param_indices[key]
        param_inits = []
        for idx in param_indices:
            mask = self._param_participance_masks[idx]
            param_init = hard_thresholding_operator(np.nansum(As[mask], axis=0) / mask.sum(), max_rank=self.init_rank)
            param_inits.append(param_init)
        self._set_matrices(param_inits, indices=param_indices)

    def _update_history(self, nll: float, key: Tuple[int, int],
                        save_nll_history: bool = True,
                        save_latent_history: bool = False) -> None:
        """
        Update optimization history (NLL and optionally latent states).

        Parameters
        ----------
        nll : float
            Current negative log-likelihood value.
        key : tuple of int
            Identifier for optimization stage.
        save_nll_history : bool, default=True
            Whether to save NLL history.
        save_latent_history : bool, default=False
            Whether to save latent variable history.
        """
        if save_nll_history:
            self._update_nll_history(nll=nll, key=key)
        if save_latent_history:
            self._update_latent_history(key)

    def _reset_history(self, key: Tuple[int, int]):
        """
        Reset optimization history for a given key.

        Parameters
        ----------
        key : tuple of int
            Identifier for optimization stage.
        """
        hist_key = self._key_2_history_key[key]
        self.nll_history_[hist_key][:] = []

    def _update_nll_history(self, nll: float, key: Tuple[int, int]):
        """
        Append new negative log-likelihood value to history.

        Parameters
        ----------
        nll : float
            Negative log-likelihood value.
        key : tuple of int
            Identifier for optimization stage.
        """
        history_key = self._key_2_history_key[key]
        self.nll_history_[history_key].append(nll)

    def _update_latent_history(self, key: Tuple[int, int]):
        """
        Append current latent space matrices to history.

        Parameters
        ----------
        key : tuple of int
            Identifier for optimization stage.
        """
        history_key = self._key_2_history_key[key]
        self.latent_history_[history_key].append(deepcopy(self.get_all_fitted_matrices_by_type()))

    def _make_key_2_history_key_dict(self) -> None:
        """Create mapping from optimization keys to history keys."""
        self._key_2_history_key = {(0, 0): "First Stage"}

    def _make_key_2_param_indices_dict(self) -> None:
        """Create mapping from optimization keys to parameter index lists."""
        self._key_2_param_indices = {(0, 0): list(range(self.n_fitted_matrices))}

    def _make_key_2_hyperparams_dict(self) -> None:
        """Initialize empty dictionary for hyperparameters per optimization key."""
        self._key_2_hyperparams_dict = dict()

    def _init_keys(self):
        """Initialize all internal key mapping dictionaries."""
        self._make_key_2_history_key_dict()
        self._make_key_2_param_indices_dict()
        self._make_key_2_hyperparams_dict()

    def _set_hyperparams(self, params_dict: Dict[str, float], key: Tuple[int, int]):
        """
        Set hyperparameter values for the given key.

        Parameters
        ----------
        params_dict : dict
            Dictionary mapping hyperparameter names to their values.
        key : tuple of int
            Optimization stage identifier.
        """
        for param_name, param_val in params_dict.items():
            old_val = self._key_2_hyperparams_dict[key][param_name]
            param_val = param_val if np.isscalar(old_val) else param_val * np.ones_like(old_val)
            self._key_2_hyperparams_dict[key][param_name] = param_val

    def _init_history(self, manager: Manager):
        """
        Initialize shared memory-based history tracking structures.

        Parameters
        ----------
        manager : multiprocessing.Manager
            Shared memory manager for inter-process synchronization.
        """
        self._shm_lock = manager.Lock()
        self.optim_info = manager.dict({'n_svd_calls': 0})
        self.nll_history_ = manager.dict({hist_key: manager.list() for hist_key in self._key_2_history_key.values()})
        self.latent_history_ = manager.dict({hist_key: manager.list() for hist_key in self._key_2_history_key.values()})

    def _validate_hyperparams(self):
        """
        Validate and initialize hyperparameters such as ranks and sigma scales.

        Raises
        ------
        AssertionError
            If initial or maximum rank constraints are violated.
        NotImplementedError
            If sigmas dimensionality is unsupported.
        """
        if self.init_rank is None:
            self.init_rank = 1.

        assert self.n_nodes_ >= self.init_rank, "Initial rank should not be greater than the number of nodes"
        if self.max_rank is not None:
            assert self.max_rank >= self.init_rank, "Initial rank should not be greater than maximum rank"
        else:
            self.max_rank = int(np.sqrt(self.n_nodes_))

        if self.sigmas is None:
            self.sigmas_ = np.ones((self.n_layers_, self.n_nodes_, self.n_nodes_), dtype=float)
        else:
            self.sigmas_ = np.array(self.sigmas, dtype=float)

        if self.sigmas_.ndim == 0:
            self.sigmas_ = self.sigmas_ * np.ones((self.n_layers_, self.n_nodes_, self.n_nodes_), dtype=float)
        elif self.sigmas_.ndim == 1:
            self.sigmas_ = self.sigmas_[:, None, None] * np.ones((1, self.n_nodes_, self.n_nodes_))
        elif self.sigmas_.ndim == 2:
            assert self.sigmas_.shape == (self.n_nodes_, self.n_nodes_)
            self.sigmas_ = np.stack([self.sigmas_] * self.n_layers_)
        elif self.sigmas_.ndim == 3:
            assert self.sigmas_.shape == (self.n_layers_, self.n_nodes_, self.n_nodes_)
        else:
            raise NotImplementedError("sigmas should be a scalar, 1d, 2d, or 3d array.")

    def _compute_projected_map(self, idx: int, lr: float, key: Tuple[int, int]) -> Callable:
        """Placeholder for projected map computation (to be implemented in subclass)."""
        pass

    def _update_matrix(self, idx, lr: float, sample_grads: np.array, key: Tuple[int, int]):
        """
        Update a single parameter matrix via gradient step and projection.

        Parameters
        ----------
        idx : int
            Index of the parameter matrix to update.
        lr : float
            Learning rate.
        sample_grads : np.ndarray
            Gradient tensor of the same shape as observed data.
        key : tuple of int
            Optimization stage identifier.
        """
        projected_map = self._compute_projected_map(idx, lr=lr, key=key)
        mask = self._param_participance_masks[idx]
        matrix_grad = np.nansum(sample_grads[mask], axis=0)
        cur_matrix = self.get_all_fitted_matrices()[idx]
        upd_matrix = projected_map(cur_matrix - lr / mask.sum() * matrix_grad)
        self._set_matrix(upd_matrix, idx)

    @property
    def n_fitted_matrices(self) -> int:
        """int: Total number of fitted matrices (shared + individual)."""
        return 1 + self.n_layers_

    def _pre_fit(self):
        """Perform pre-fitting initialization including parameters, keys, and history."""
        self._init_param_matrices(np.zeros((self.n_fitted_matrices, self.n_nodes_, self.n_nodes_)))
        self._validate_hyperparams()
        self._init_keys()
        self._init_history(manager=Manager())

    def _optimization_step(self, param_indices, key: Tuple[int, int], As, lr: float, iteration: int,
                           verbose: bool, tol: float,
                           save_latent_history: bool, save_nll_history: bool) -> float:
        """
        Execute one optimization iteration over a subset of parameters.

        Parameters
        ----------
        param_indices : list of int
            Indices of parameters to optimize.
        key : tuple of int
            Optimization stage identifier.
        As : np.ndarray
            Input adjacency tensors.
        lr : float
            Learning rate.
        iteration : int
            Current iteration number.
        verbose : bool
            Whether to print progress.
        tol : float
            Tolerance for rounding output.
        save_latent_history : bool
            Whether to store latent variable history.
        save_nll_history : bool
            Whether to store NLL history.

        Returns
        -------
        float
            Negative log-likelihood after the update step.
        """
        for idx in param_indices:
            sample_grads = self._nll_sample_grads(As)
            self._update_matrix(idx, lr=lr, sample_grads=sample_grads, key=key)

        obs_mask = self._param_participance_masks[param_indices].any(0)
        nll = self._compute_nll(As, obs_mask=obs_mask)

        with self._shm_lock:
            self._update_history(nll, key=key,
                                 save_latent_history=save_latent_history, save_nll_history=save_nll_history)

        if verbose:
            nll = np.round(nll, 1 - int(np.log10(tol))) if tol > 0 else nll
            print(f"{self._key_2_history_key[key]}, Iteration {iteration}, NLL {nll}")
        return nll

    def _optimize_params(self, key: Tuple[int, int], As, lr: float,
                         max_iter: int, tol: float, patience,
                         verbose: bool, verbose_interval: int,
                         save_latent_history: bool, save_nll_history: bool,
                         refit: bool):
        """
        Optimize model parameters via iterative updates and early stopping.

        Parameters
        ----------
        key : tuple of int
            Optimization stage identifier.
        As : np.ndarray
            Observed adjacency tensors.
        lr : float
            Learning rate.
        max_iter : int
            Maximum number of iterations.
        tol : float
            Stopping tolerance for early stopping.
        patience : int
            Number of iterations with no improvement before stopping.
        verbose : bool
            Whether to print progress.
        verbose_interval : int
            Interval (in iterations) for printing updates.
        save_latent_history : bool
            Whether to save latent variables at each step.
        save_nll_history : bool
            Whether to save NLL values.
        refit : bool
            Whether to refit parameters after optimization.
        """
        self._reset_history(key=key)
        param_indices = self._key_2_param_indices[key]
        self._optimized_params_init(As, key=key)
        early_stopper = EarlyStopper(patience=patience, higher_better=False, rtol=tol)
        for iteration in range(max_iter):
            nll = self._optimization_step(param_indices, key=key, As=As, lr=lr, iteration=iteration, tol=tol,
                                          verbose=verbose and iteration % verbose_interval == 0,
                                          save_latent_history=save_latent_history, save_nll_history=save_nll_history)

            if early_stopper.stop_check(nll):
                with self._shm_lock:
                    self.optim_info["n_svd_calls"] += (iteration + 1) * len(param_indices)
                if verbose:
                    print(f"Early stop activated on iteration {iteration}!\n")
                break

        if refit:
            self.refit(As, refit_matrix_indices=param_indices)

    def _fit(self, As, **optim_kwargs):
        """
        Full model fitting routine combining initialization and optimization.

        Parameters
        ----------
        As : np.ndarray
            Input adjacency tensors.
        **optim_kwargs
            Optimization keyword arguments passed to `_optimize_params`.
        """
        self._pre_fit()
        self._optimize_params(As=As, key=(0, 0), **optim_kwargs)

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
        triu_mask = np.triu(np.ones((self.n_nodes_, self.n_nodes_)), k=0 if self.loops_allowed else 1)
        Ps = self.compute_expected_adjacency()
        if self.edge_distrib == "normal":
            full_nll = (As - Ps) ** 2 / (2. * self.sigmas_ ** 2)
        else:
            Ps = np.clip(Ps, eps, 1. - eps)
            full_nll = -As * np.log(Ps) - (1. - As) * np.log(1. - Ps)
        return np.nansum(full_nll[obs_mask] * triu_mask)

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
        """
        Retrieve history of parameter estimation errors across iterations.

        Parameters
        ----------
        true_params : tuple[np.array]
            Ground truth parameters.
        relative_errors : bool, default=True
            Whether to compute relative errors.

        Returns
        -------
        list of dict
            Error values for each iteration.
        """
        assert self.latent_history_, "Set save_latent_history == True during fit to access latent_history!"
        assert len(true_params) == len(self._get_all_fitted_parameter_names()), \
            f"Length of true_params should be equal to {len(self._get_all_fitted_parameter_names())}"
        err_hist = []
        for estimate_params in self.latent_history_:
            iteration_errs = make_error_report(true_params, estimate_params, relative_errors=relative_errors)
            err_hist.append(iteration_errs)
        return err_hist

    def plot_nll_history(self, n_cols: int = 4, fontsize: int = 14, figsize=(12, 20)):
        """
        Plot negative log-likelihood history across optimization stages.

        Parameters
        ----------
        n_cols : int, default=4
            Number of subplot columns.
        fontsize : int, default=14
            Font size for labels and titles.
        figsize : tuple, default=(12, 20)
            Figure size.
        """
        n_cols = min(n_cols, len(self.nll_history_))
        n_rows = int(np.ceil(len(self.nll_history_) / n_cols))
        fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=figsize)
        axs = np.array(axs).flatten()
        for ax, (stage, nll_hist) in zip(axs, self.nll_history_.items()):
            ax.plot(nll_hist)
            ax.set_xlabel("Iteration", fontsize=fontsize)
            ax.set_ylabel("Negative Loglikelihood", fontsize=fontsize)
            ax.set_title(f"{stage}", fontsize=fontsize)
            ax.tick_params(axis='both', labelsize=fontsize)
        for idx in range(len(self.nll_history_), n_cols * n_rows):
            fig.delaxes(axs[idx])  # Remove extra axes
        plt.tight_layout()
        plt.show()

    def plot_error_history(self, *true_params, relative_errors=True, labels=None, fontsize=14,
                           print_final_errors=True):
        """
        Plot evolution of parameter estimation errors over iterations.

        Parameters
        ----------
        *true_params : list of np.ndarray
            Ground truth parameters.
        relative_errors : bool, default=True
            Whether to plot relative errors.
        labels : list of str, optional
            Custom labels for parameters.
        fontsize : int, default=14
            Font size for plot.
        print_final_errors : bool, default=True
            Whether to print final iteration errors.
        """
        if labels is None:
            labels = self._get_all_fitted_parameter_names()
        else:
            assert len(true_params) == len(labels), \
                "labels should be of the same length as the length of true_params"

        err_hist = self.get_error_history(*true_params, relative_errors=relative_errors)
        if print_final_errors:
            print("Final errors:", np.round(err_hist[-1], 2))
        plt.figure(figsize=(12, 8))
        plt.plot(err_hist, label=labels)
        plt.legend(fontsize=fontsize)
        plt.xlabel("Iteration", fontsize=fontsize)
        plt.ylabel("Error", fontsize=fontsize)
        plt.title("Relative Frobenius Errors", fontsize=fontsize)


class BaseMultiNeSSRefinedCV(BaseMultiNeSSRefined):
    """
    Cross-validated refinement class for MultiNeSS models.

    This class extends `BaseMultiNeSSRefined` by adding cross-validation (CV) procedures
    for hyperparameter selection. It supports k-fold CV or a single hold-out test split,
    and selects optimal parameters based on the mean negative log-likelihood (NLL).

    Parameters
    ----------
    param_grid : dict of {str: list}, optional
        Dictionary defining the hyperparameter grid to explore.
    cv_folds : int, optional
        Number of cross-validation folds. If None, a single train/test split is used.
    test_prop : float, default=0.1
        Proportion of data used for the test set in a single split.
    fix_layer_split : bool, default=False
        Whether to fix the split of layers across folds.
    use_1se_rule : bool, default=False
        If True, selects the most parsimonious model within one standard error of the best mean NLL.
    random_seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, param_grid: Dict[str, List] = None, cv_folds=None, test_prop=0.1,
                 fix_layer_split: bool = False, use_1se_rule=False, random_seed: int = None, **kwargs):
        super().__init__(**kwargs)
        self.cv_folds = cv_folds
        self.test_prop = test_prop
        if param_grid is None:
            self.param_grid = param_grid
        else:
            self.param_grid = dict(sorted(param_grid.items(), key=lambda kv: kv[0], reverse=True))
        self.use_1se_rule = use_1se_rule
        self.fix_layer_split = fix_layer_split
        self.random_seed = random_seed

    def _make_key_2_hyperparams_grid_dict(self) -> None:
        """
        This method is typically called during setup to prepare
        storage for layer-specific hyperparameter grids.
        """
        self._key_2_hyperparams_grid = dict()

    def _init_keys(self):
        """
        Initialize model keys and corresponding hyperparameter grids.

        Calls the parent `_init_keys` and then creates the
        `_key_2_hyperparams_grid` dictionary.
        """
        super()._init_keys()
        self._make_key_2_hyperparams_grid_dict()

    def _init_history(self, manager: Manager):
        """
        Initialize multiprocessing-safe containers for cross-validation results.

        Parameters
        ----------
        manager : multiprocessing.Manager
            Shared-memory manager used to create thread-safe dictionaries and lists.
        """
        super()._init_history(manager)
        self._cv_results = manager.dict(
            {key: manager.dict({params: manager.list() for params in product(*param_grid_dict.values())})
             for key, param_grid_dict in self._key_2_hyperparams_grid.items()}
        )
        self._best_params_dict = manager.dict()

    def _set_best_hyperparams(self, key: Tuple[int, int]) -> None:
        """
        Determine and store the best hyperparameters for a given key.

        Parameters
        ----------
        key : tuple of int
            Identifier for the network-layer combination being optimized.
        """
        res_dict = self._cv_results[key]
        params, param_nlls = np.stack(list(res_dict.keys())), np.stack(list(res_dict.values()))
        optim_nll_idx = np.argmin(param_nlls.mean(1))
        if self.use_1se_rule and self.cv_folds > 1:
            optim_param_nll_std = param_nlls.std(1, ddof=1)[optim_nll_idx]
            param_nll_means = param_nlls.mean(1)
            param_nll_1se_ub = param_nll_means[optim_nll_idx] + optim_param_nll_std / np.sqrt(self.cv_folds)
            best_params = max(params[param_nll_means <= param_nll_1se_ub].tolist())
        else:
            best_params = params[optim_nll_idx]

        with self._shm_lock:
            self._best_params_dict[key] = dict(zip(self._key_2_hyperparams_grid[key].keys(), best_params))
        self._set_hyperparams(params_dict=self._best_params_dict[key], key=key)

    def _cv_hyperparams(self, As, fit_fun: Callable, comp_nll_fun: Callable, key: Tuple[int, int],
                        verbose=False):
        """
        Perform cross-validation for a given key and hyperparameter grid.

        Parameters
        ----------
        As : list or np.ndarray
            Input adjacency matrices or network representations.
        fit_fun : callable
            Function to fit the model on the training data.
        comp_nll_fun : callable
            Function to compute the negative log-likelihood on the test data.
        key : tuple of int
            Identifier for the network-layer combination.
        verbose : bool, default=False
            Whether to print progress and intermediate results.
        """
        param_grid_dict = self._key_2_hyperparams_grid[key]
        param_name_tuples = list(param_grid_dict.keys())

        tts = MultipleNetworkTrainTestSplitter(loops_allowed=self.loops_allowed,
                                               fix_layer_split=self.fix_layer_split, random_seed=self.random_seed)
        if self.cv_folds is None:
            fold_gen = [tts.train_test_split(As, test_prop=self.test_prop)]
        else:
            fold_gen = tts.kfold_split(As, n_splits=self.cv_folds)

        for fold, (As_train, As_test) in enumerate(fold_gen):
            if verbose:
                print(f"#### CV Fold {fold + 1} ####\n")

            for params in product(*param_grid_dict.values()):
                params_dict = dict(zip(param_name_tuples, params))
                if verbose:
                    print(f"\nStart working on {params_dict}", end="\n\n")
                self._set_hyperparams(params_dict=params_dict, key=key)
                fit_fun(As=As_train)
                nll = comp_nll_fun(As_test)
                with self._shm_lock:
                    self._cv_results[key][params].append(nll)

        self._set_best_hyperparams(key=key)
        if verbose:
            print("\n#### Fitting on full data with best parameters! ####", end="\n\n")
        fit_fun(As=As)

    def get_cv_results(self):
        """
        Retrieve the stored cross-validation results.

        Returns
        -------
        dict
            Nested dictionary mapping each key to its parameter combinations and
            the list of NLL values across folds.
        """
        assert hasattr(self, "_cv_results"), "Model was not fitted!"
        return {key: {param: list(nll_list) for param, nll_list in param_dict.items()}
                for key, param_dict in self._cv_results.items()}

    def plot_cv_results(self, log_params=True, fontsize=12):
        """
        Plot the cross-validation results for all keys.

        Parameters
        ----------
        log_params : bool, default=True
            Whether to display hyperparameter values on a log10 scale.
        fontsize : int, default=12
            Font size for plot labels and titles.
        """
        cv_res = sorted(self.get_cv_results().items(), key=lambda kv: kv[0])
        fig, axs = plt.subplots(1, len(cv_res), figsize=(20, 6))
        if len(cv_res) == 1:
            axs = [axs]
        for ax, (key, key_res) in zip(axs, cv_res):
            hyperparam_2_grid = self._key_2_hyperparams_grid[key]
            param_names = hyperparam_2_grid.keys()
            ax.set_title(self._key_2_history_key[key], fontsize=fontsize + 3)
            if len(param_names) == 1:
                param_vals = np.hstack(list(key_res.keys()))
                param_vals = np.log10(param_vals).round(2) if log_params else param_vals.round(2)
                param_name = r'$\lambda$' if list(param_names)[0].startswith("lmbda") else r'$\alpha$'
                cv_errs_over_folds = np.stack(list(key_res.values()))
                for fold in range(cv_errs_over_folds.shape[1]):
                    ax.plot(param_vals, cv_errs_over_folds[:, fold], label=f"Fold={fold + 1}", marker="o")
                ax.set_xticks(param_vals, param_vals, rotation=30, fontsize=fontsize)
                ax.set_xlabel(r'$\log_{10}$' + f'({param_name})' if log_params else param_name, fontsize=fontsize + 3)
                ax.set_ylabel("NLL", fontsize=fontsize + 3)
                ax.legend(fontsize=fontsize)
            else:
                lmbda_grid, alpha_grid = hyperparam_2_grid.values()
                heatmap_vals = [[np.mean(key_res[(lmbda, alpha)]) for lmbda in lmbda_grid] for alpha in alpha_grid]
                best_lmbda, best_alpha = self.get_best_params_dict()[key].values()
                col, row = list(lmbda_grid).index(best_lmbda), list(alpha_grid).index(best_alpha)
                sns.heatmap(heatmap_vals, ax=ax)
                ax.text(col + 0.5, row + 0.5, "X", c="red", fontsize=fontsize)

                ax.set_xlabel(r'$\lambda$', fontsize=fontsize + 3)
                ax.set_ylabel(r'$\alpha$', fontsize=fontsize + 3)
                ax.set_xticklabels(lmbda_grid.round(2), fontsize=fontsize, rotation=30)
                ax.set_yticklabels(alpha_grid.round(2), fontsize=fontsize)
        plt.show()

    def get_best_params_dict(self):
        """
        Retrieve the best hyperparameters for each key.

        Returns
        -------
        dict
            Mapping from keys to the best parameter combinations selected by CV.
        """
        assert hasattr(self, "_best_params_dict"), "Model was not fitted!"
        return dict(self._best_params_dict)

    def _optimize_params(self, key: Tuple[int, int], As, **optim_params):
        """
        Optimize parameters for a given key, with optional cross-validation.

        Parameters
        ----------
        key : tuple of int
            Identifier for the network-layer combination.
        As : list or np.ndarray
            Input adjacency matrices or network representations.
        **optim_params : dict
            Additional optimization parameters such as verbosity or stopping criteria.
        """
        param_indices = self._key_2_param_indices[key]
        obs_mask = self._param_participance_masks[param_indices].any(0)
        comp_nll_fun = partial(super()._compute_nll, obs_mask=obs_mask)
        fit_fun = partial(super()._optimize_params, key=key, **optim_params)
        if self._key_2_hyperparams_grid[key]:
            self._cv_hyperparams(As, fit_fun=fit_fun, comp_nll_fun=comp_nll_fun, key=key,
                                 verbose=optim_params["verbose"])
        else:
            super()._optimize_params(key=key, As=As, **optim_params)


class MultiNeSS(BaseMultiNeSSRefined, BaseRefitting):
    """
    This class extends `BaseMultiNeSSRefined` by adding refitting and fixing updates
    as soft-thresholding with given parameters

    Parameters
    ----------
    lmbda : float, optional
        Global penalty scaling factor.
    alpha : float, list, or np.ndarray, optional
        Layer-specific scaling factors for penalties.
    max_rank : int, optional
        Maximum allowed rank for low-rank projection.
    init_rank : int, optional
        Initial rank used for parameter initialization.
    edge_distrib : str, default='normal'
        Edge distribution assumed ('normal' or 'bernoulli').
    sigmas : float, list, or np.ndarray, optional
        Standard deviation(s) for edges when edge_distrib='normal'.
    loops_allowed : bool, default=True
        Whether self-loops are allowed in the network.
    refit_threshold : float, default=1e-8
        Threshold for triggering refit in BaseRefitting.
    """

    def __init__(self,
                 lmbda: float = None,
                 alpha: Union[float, List, np.ndarray] = None,
                 max_rank: int = None, init_rank: int = None,
                 edge_distrib: str = "normal",
                 sigmas: Union[float, List, np.ndarray] = None,
                 loops_allowed: bool = True,
                 refit_threshold=1e-8):

        BaseMultiNeSSRefined.__init__(self, edge_distrib=edge_distrib, max_rank=max_rank, init_rank=init_rank,
                                      sigmas=sigmas, loops_allowed=loops_allowed)
        BaseRefitting.__init__(self, edge_distrib=edge_distrib, max_rank=max_rank, loops_allowed=loops_allowed,
                               refit_threshold=refit_threshold)
        self.lmbda = lmbda
        self.alpha = alpha

    def _compute_penalty_coef(self, idx: int, key: Tuple[int, int]) -> float:
        """
        Compute the penalty coefficient for a specific parameter index.

        Parameters
        ----------
        idx : int
            Index of the parameter matrix.
        key : tuple of int
            Network-layer key for selecting hyperparameters.

        Returns
        -------
        float
            Penalty coefficient applied to the soft-thresholding operator.
        """
        hyperparam_dict = self._key_2_hyperparams_dict[key]
        lmbda, alpha = hyperparam_dict["lmbda"], hyperparam_dict["alpha"]
        return lmbda if idx == 0 else lmbda * alpha[idx - 1]

    def _compute_projected_map(self, idx: int, lr, key: Tuple[int, int]):
        """
        Generate a projected update function (soft-thresholding) for a parameter.

        Parameters
        ----------
        idx : int
            Index of the parameter matrix.
        lr : float
            Learning rate.
        key : tuple of int
            Network-layer key.

        Returns
        -------
        callable
            A function that applies soft-thresholding with appropriate threshold.
        """
        mask = self._param_participance_masks[idx]
        penalty = self._compute_penalty_coef(idx=idx, key=key)
        return partial(soft_thresholding_operator, threshold=penalty * lr / mask.sum(), max_rank=self.max_rank)

    def _make_key_2_hyperparams_dict(self):
        """
        Initialize the mapping from network-layer key to hyperparameters.

        Sets default values for `lmbda` and `alpha` if not provided.
        """
        lmbda = 2. * np.sqrt(self.n_layers_ * self.n_nodes_) if self.lmbda is None else self.lmbda
        if self.alpha is None:
            alpha = np.ones(self.n_layers_) / np.sqrt(self.n_layers_)
        else:
            alpha = if_scalar_or_given_length_array(self.alpha, length=self.n_layers_, name="alpha")
        self._key_2_hyperparams_dict = {(0, 0): {"lmbda": lmbda, "alpha": alpha}}

    def fit(self, As: List[np.array],
            lr: float = 1.,
            max_iter: int = 100, tol=1e-5, patience=10,
            refit=True,
            verbose=True, verbose_interval=1,
            save_latent_history=False, save_nll_history=True):
        """
        Fit the MultiNeSS model to a list of adjacency matrices.

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
            Number of iterations for early stopping patience.
        refit : bool, default=True
            Whether to perform refitting after convergence.
        verbose : bool, default=True
            Whether to print progress messages.
        verbose_interval : int, default=1
            Interval between printing verbose messages.
        save_latent_history : bool, default=False
            Whether to save latent matrix history.
        save_nll_history : bool, default=True
            Whether to save negative log-likelihood history.

        Returns
        -------
        self
            Fitted MultiNeSS object.
        """
        As = self._validate_input(As)
        self._fit(As, lr=lr,
                  max_iter=max_iter, tol=tol, patience=patience,
                  refit=refit,
                  verbose=verbose, verbose_interval=verbose_interval,
                  save_latent_history=save_latent_history, save_nll_history=save_nll_history)
        return self


class MultiNeSSCV(MultiNeSS, BaseMultiNeSSRefinedCV):
    """
    Cross-validated MultiNeSS model.

    Extends `MultiNeSS` with hyperparameter selection via cross-validation.
    """

    def __init__(self, param_grid: Dict[str, List[float]] = None, cv_folds=3, test_prop=0.1,
                 lmbda: float = None,
                 alpha: Union[float, np.ndarray, list] = None,
                 edge_distrib: str = "normal",
                 max_rank: int = None, init_rank: int = None,
                 sigmas: Union[float, List, np.ndarray] = None,
                 loops_allowed: bool = True,
                 refit_threshold: float = 1e-8,
                 fix_layer_split: bool = False,
                 use_1se_rule: bool = False,
                 random_seed: int = None):
        """
        Initialize a MultiNeSS model with cross-validation.

        Parameters
        ----------
        param_grid : dict of {str: list of float}, optional
            Grid of hyperparameters for CV.
        cv_folds : int, default=3
            Number of CV folds.
        test_prop : float, default=0.1
            Test set proportion for single split.
        lmbda : float, optional
            Global penalty factor.
        alpha : float, array-like, optional
            Layer-specific penalty factors.
        edge_distrib : str, default='normal'
            Edge distribution.
        max_rank : int, optional
            Maximum rank for projection.
        init_rank : int, optional
            Initial rank for initialization.
        sigmas : float or array-like, optional
            Edge standard deviations for 'normal' distribution.
        loops_allowed : bool, default=True
            Allow self-loops.
        refit_threshold : float, default=1e-8
            Threshold for refitting.
        fix_layer_split : bool, default=False
            Fix layer split across folds.
        use_1se_rule : bool, default=False
            Use 1-SE rule for selecting parsimonious model.
        random_seed : int, optional
            Random seed for reproducibility.
        """
        BaseMultiNeSSRefinedCV.__init__(self, param_grid=param_grid, cv_folds=cv_folds, test_prop=test_prop,
                                        loops_allowed=loops_allowed,
                                        fix_layer_split=fix_layer_split,
                                        use_1se_rule=use_1se_rule,
                                        random_seed=random_seed)

        MultiNeSS.__init__(self, lmbda=lmbda, alpha=alpha, edge_distrib=edge_distrib,
                           max_rank=max_rank, init_rank=init_rank,
                           sigmas=sigmas, loops_allowed=loops_allowed, refit_threshold=refit_threshold)

    def _make_key_2_hyperparams_grid_dict(self) -> None:
        """
        Create a default or user-specified hyperparameter grid for cross-validation.
        """
        if self.param_grid is None:
            scale = np.sqrt(self.n_nodes_ * self.n_layers_)
            const_range = np.array([0.03, 0.1, 0.3, 1, 3, 10])
            param_grid = {"lmbda": scale * const_range} if self.param_grid is None else self.param_grid
            self._key_2_hyperparams_grid = {(0, 0): param_grid}
        else:
            self._key_2_hyperparams_grid = {(0, 0): self.param_grid}


class BaseOracleMultiNeSS(BaseMultiNeSSRefined):
    """
    Oracle MultiNeSS with rank thresholding.

    Parameters
    ----------
    Inherits all parameters from BaseMultiNeSSRefined.
    """

    def _compute_threshold_rank(self, idx: int, key: Tuple[int, int]) -> int:
        """
        Compute the threshold rank for hard-thresholding operator.

        Parameters
        ----------
        idx : int
            Parameter matrix index.
        key : tuple of int
            Network-layer key.

        Returns
        -------
        int
            Threshold rank.
        """
        pass

    def _compute_projected_map(self, idx: int, lr, key: Tuple[int, int]):
        """
        Generate a projected update function (hard-thresholding) for a parameter.

        Parameters
        ----------
        idx : int
            Index of the parameter matrix.
        lr : float
            Learning rate.
        key : tuple of int
            Network-layer key.

        Returns
        -------
        callable
            Function applying hard-thresholding with computed max rank.
        """
        rank = self._compute_threshold_rank(idx=idx, key=key)
        return partial(hard_thresholding_operator, max_rank=rank)

    def _optimized_params_init(self, As, key: Tuple[int, int]) -> None:
        """
        Initialize parameter matrices using hard-thresholding with threshold ranks.

        Parameters
        ----------
        As : list of np.ndarray
            Observed adjacency matrices.
        key : tuple of int
            Network-layer key.
        """
        param_indices = self._key_2_param_indices[key]
        param_inits = []
        for idx in param_indices:
            mask = self._param_participance_masks[idx]
            max_rank = self._compute_threshold_rank(idx=idx, key=key)
            param_init = hard_thresholding_operator(np.nansum(As[mask], axis=0) / mask.sum(), max_rank=max_rank)
            param_inits.append(param_init)
        self._set_matrices(param_inits, indices=param_indices)

    def fit(self, As: List[np.array],
            lr: float = 1.,
            max_iter: int = 100, tol=1e-5, patience=10,
            verbose=True, verbose_interval=1,
            save_latent_history=False, save_nll_history=True):
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
        As = self._validate_input(As)
        self._fit(As, lr=lr,
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
        Edge distribution ('normal' or 'bernoulli').
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
        d_shared = if_scalar_or_given_length_array(self.d_shared, length=1, name="d_shared")
        d_individs = if_scalar_or_given_length_array(self.d_individs, length=self.n_layers_, name="d_individs")
        self._key_2_hyperparams_dict = {(0, 0): {"d_shared": d_shared, "d_individs": d_individs}}

    def _compute_threshold_rank(self, idx: int, key: Tuple[int, int]) -> int:
        """
        Compute the threshold rank for the hard-thresholding operator.

        Parameters
        ----------
        idx : int
            Index of the parameter matrix (0 for shared component, >0 for individual layers).
        key : tuple of int
            Network-layer key.

        Returns
        -------
        int
            Rank threshold for hard-thresholding.
        """
        hyperparam_dict = self._key_2_hyperparams_dict[key]
        d_shared, d_individs = hyperparam_dict["d_shared"], hyperparam_dict["d_individs"]
        return d_shared if idx == 0 else d_individs[idx - 1]
