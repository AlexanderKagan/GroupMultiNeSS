import numpy as np
from typing import List, Dict, Union, Tuple
from collections import Counter
from functools import partial
from joblib import Parallel, delayed
from warnings import warn

from .multiness import BaseMultiNeSS, BaseMultiNeSSRefined, BaseMultiNeSSRefinedCV, BaseOracleMultiNeSS
from .base import BaseRefitting
from .utils import if_scalar_or_given_length_array, fill_diagonals, \
    soft_thresholding_operator



class BaseGroupMultiNeSS(BaseMultiNeSS):
    """
    Base class for group-structured MultiNeSS models.

    Handles shared, group-specific, and individual latent spaces for network layers.

    Parameters
    ----------
    group_indices : List[int]
        List of group assignments for each network layer.
    edge_distrib : str, default='normal'
        Distribution of edges ('normal' or 'bernoulli').
    loops_allowed : bool, default=True
        Whether self-loops are allowed in the network.
    """

    def __init__(self, group_indices: List[int],
                 edge_distrib: str = "normal",
                 loops_allowed: bool = True):
        """
        Initialize BaseGroupMultiNeSS with group assignments.
        """
        super().__init__(edge_distrib=edge_distrib, loops_allowed=loops_allowed)
        self.group_indices = group_indices

    def compute_latent_spaces(self):
        """
        Compute the full latent space by summing shared, group-specific, and individual components.

        Returns
        -------
        np.ndarray
            The combined latent space matrix.
        """
        ls = self.get_shared_latent_space() + self.get_group_latent_spaces()[self.group_indices] + \
            self.get_individual_latent_spaces()
        if not self.loops_allowed:
            fill_diagonals(ls, val=0., inplace=True)
        return ls

    def get_shared_latent_space(self):
        """
        Get the shared latent space.

        Returns
        -------
        np.ndarray
            Shared component matrix.
        """
        return self.get_all_fitted_matrices()[0]

    def get_group_latent_spaces(self):
        """
        Get the group-specific latent spaces.

        Returns
        -------
        np.ndarray
            Group component matrices.
        """
        return self.get_all_fitted_matrices()[1: self.n_groups_ + 1]

    def get_individual_latent_spaces(self):
        """
        Get the individual latent spaces.

        Returns
        -------
        np.ndarray
            Individual component matrices.
        """
        return self.get_all_fitted_matrices()[self.n_groups_ + 1:]

    def get_all_fitted_matrices_by_type(self):
        """
        Return all fitted matrices separated by type: shared, group, individual.

        Returns
        -------
        list
            List of matrices by type.
        """
        return [self.get_shared_latent_space(), self.get_group_latent_spaces(), self.get_individual_latent_spaces()]

    @staticmethod
    def _get_all_fitted_parameter_names():
        """
        Names of the fitted parameter groups.

        Returns
        -------
        list
            List of strings representing component types.
        """
        return ["Shared component", "Group components", "Individual components"]

    def _validate_input(self, As: List[np.array]):
        """
        Validate input data and group assignments.

        Parameters
        ----------
        As : List[np.ndarray]
            List of network adjacency matrices.

        Returns
        -------
        np.ndarray
            Validated adjacency matrices.
        """
        if np.any([group_size == 1 for group_size in Counter(self.group_indices).values()]):
            warn("Group latent space is unidentifiable for groups of size 1.")
        self.unique_groups_ = np.sort(np.unique(self.group_indices))
        self.group_sizes_ = np.array([(self.group_indices == group).sum() for group in self.unique_groups_])
        self.n_groups_ = len(self.unique_groups_)
        assert np.all(np.sort(self.unique_groups_) == np.arange(len(self.unique_groups_))), \
            "Groups should be named from 0 to K-1, where K is the number of unique groups."
        As = super()._validate_input(As)
        assert len(self.group_indices) == len(As), "Length of group_indices should be equal to the number of layers"
        return As


class BaseGroupMultiNeSSRefined(BaseGroupMultiNeSS, BaseMultiNeSSRefined):
    """
    Refined group-structured MultiNeSS model.

    Inherits from BaseGroupMultiNeSS and BaseMultiNeSSRefined.
    Handles optimization over shared, group-specific, and individual components.

    Parameters
    ----------
    group_indices : List[int]
        List of group assignments for each network layer.
    edge_distrib : str, default='normal'
        Edge distribution.
    max_rank : int, optional
        Maximum rank for low-rank components.
    init_rank : int, optional
        Initial rank for low-rank components.
    sigmas : float, list, or np.ndarray, optional
        Noise standard deviation(s).
    loops_allowed : bool, default=True
        Whether self-loops are allowed.
    n_jobs : int, optional
        Number of parallel jobs for optimization.
    """

    def __init__(self, group_indices: List[int],
                 edge_distrib: str = "normal",
                 max_rank: int = None, init_rank: int = None,
                 sigmas: Union[float, List, np.ndarray] = None,
                 loops_allowed: bool = True,
                 n_jobs: int = None):
        """
        Initialize BaseGroupMultiNeSSRefined.
        """
        BaseMultiNeSSRefined.__init__(self, edge_distrib=edge_distrib, loops_allowed=loops_allowed,
                                      max_rank=max_rank, init_rank=init_rank, sigmas=sigmas)
        self.group_indices = group_indices
        self.n_jobs = n_jobs

    @property
    def n_fitted_matrices(self) -> int:
        """
        Total number of fitted matrices (shared + group + individual).

        Returns
        -------
        int
            Number of fitted matrices.
        """
        return 1 + self.n_layers_ + self.n_groups_

    def _get_group_dependent_matrix_indices(self, group) -> List:
        """
        Get indices of matrices dependent on a specific group.

        Parameters
        ----------
        group : int
            Group index.

        Returns
        -------
        list
            List of matrix indices for the group.
        """
        individs_within_group_indices = 1 + self.n_groups_ + np.arange(self.n_layers_)[self.group_indices == group]
        return [1 + group, *individs_within_group_indices]

    def _get_param_participance_masks(self):
        """
        Get boolean masks indicating parameter participation for each component.

        Returns
        -------
        np.ndarray
            Masks array of shape (n_params, n_layers).
        """
        group_masks = [self.group_indices == group for group in self.unique_groups_]
        return np.stack([np.ones(self.n_layers_, dtype=bool),
                         *group_masks,
                         *np.eye(self.n_layers_, dtype=bool)])

    def _make_key_2_history_key_dict(self):
        """
        Map optimization keys to human-readable history keys.
        """
        self._key_2_history_key = {**{(0, group): f"Stage 1, Group {group}" for group in self.unique_groups_},
                                   **{(1, 0): "Stage 2"}}

    def _make_key_2_param_indices_dict(self):
        """
        Map optimization keys to parameter indices.
        """
        self._key_2_param_indices = {**{(0, group): self._get_group_dependent_matrix_indices(group)
                                        for group in self.unique_groups_},
                                     **{(1, 0): list(range(1 + self.n_groups_))}}

    def _update_latent_history(self, key: Tuple[int, int]) -> None:
        """
        Update stored latent history for a given key.

        Parameters
        ----------
        key : tuple
            Optimization stage and group key.
        """
        stage, group = key
        history_key = self._key_2_history_key[key]
        if stage == 0:
            self.latent_history_[history_key].append(
                (self.get_group_latent_spaces()[group].copy(),
                 self.get_individual_latent_spaces()[self.group_indices == group].copy()))
        else:
            self.latent_history_[history_key].append((self.get_shared_latent_space(), self.get_group_latent_spaces()))

    def _fit(self, As, lr: Union[float, Tuple[float, float]] = 1., **optim_kwargs):
        """
        Fit the model using two-stage optimization for groups and shared components.

        Parameters
        ----------
        As : np.ndarray
            Input adjacency matrices.
        lr : float or tuple of float, default=1.
            Learning rates for stage 1 and 2.
        """
        self._pre_fit()
        lr = if_scalar_or_given_length_array(lr, length=2, name="lr")
        Parallel(n_jobs=self.n_jobs)(
            [delayed(self._optimize_params)(key=(0, group),
                                            As=As, lr=lr[0],
                                            **optim_kwargs)
             for group in self.unique_groups_])
        self._optimize_params(key=(1, 0), As=As, lr=lr[1], **optim_kwargs)


class GroupMultiNeSS(BaseGroupMultiNeSSRefined, BaseRefitting):
    """
    Group-structured MultiNeSS model with refitting capability.

    Parameters
    ----------
    group_indices : List[int]
        List of group assignments for each network layer.
    lmbda1, lmbda2 : float or array-like
        Regularization parameters for stage 1 (group) and stage 2 (shared) components.
    alpha1, alpha2 : float or array-like
        Weighting parameters for stage 1 (group) and stage 2 (shared) components.
    edge_distrib : str, default='normal'
        Edge distribution.
    max_rank, init_rank : int, optional
        Maximum and initial ranks for low-rank components.
    sigmas : float, list, or np.ndarray, optional
        Noise standard deviation(s).
    loops_allowed : bool, default=True
        Whether self-loops are allowed.
    n_jobs : int, optional
        Number of parallel jobs for optimization.
    refit_threshold : float, default=1e-8
        Threshold for refitting components.
    """

    def __init__(self, group_indices: List[int],
                 lmbda1: Union[float, np.ndarray, list] = None,
                 lmbda2: float = None,
                 alpha1: Union[float, np.ndarray, list] = None,
                 alpha2: Union[float, np.ndarray, list] = None,
                 edge_distrib: str = "normal",
                 max_rank: int = None, init_rank: int = None,
                 sigmas: Union[float, List, np.ndarray] = None,
                 loops_allowed: bool = True,
                 n_jobs: int = None,
                 refit_threshold: float = 1e-8):
        """
        Initialize GroupMultiNeSS model.
        """
        BaseGroupMultiNeSSRefined.__init__(self, group_indices=group_indices, edge_distrib=edge_distrib,
                                           max_rank=max_rank, init_rank=init_rank, sigmas=sigmas,
                                           loops_allowed=loops_allowed, n_jobs=n_jobs)
        BaseRefitting.__init__(self, edge_distrib=edge_distrib, max_rank=max_rank, loops_allowed=loops_allowed,
                               refit_threshold=refit_threshold)
        self.lmbda1 = lmbda1
        self.lmbda2 = lmbda2
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def _compute_projected_map(self, idx: int, lr, key: Tuple[int, int]):
        """
        Compute projected map for soft-thresholding operator.

        Parameters
        ----------
        idx : int
            Index of the parameter.
        lr : float
            Learning rate.
        key : tuple
            Optimization stage and group key.

        Returns
        -------
        partial
            Soft-thresholding operator with scaled threshold.
        """
        mask = self._param_participance_masks[idx]
        penalty = self._compute_penalty_coef(idx=idx, key=key)
        return partial(soft_thresholding_operator, threshold=penalty * lr / mask.sum(), max_rank=self.max_rank)

    def _compute_penalty_coef(self, idx: int, key: Tuple[int, int]) -> float:
        """
        Compute penalty coefficient for a given parameter and stage.

        Parameters
        ----------
        idx : int
            Parameter index.
        key : tuple
            Stage and group key.

        Returns
        -------
        float
            Penalty coefficient.
        """
        hyperparam_dict = self._key_2_hyperparams_dict[key]
        stage, group = key
        lmbda, alpha = hyperparam_dict[f"lmbda{stage + 1}"], hyperparam_dict[f"alpha{stage + 1}"]
        if stage == 0:
            if idx == group + 1:
                return lmbda
            else:
                idx_in_group = np.sum(self.group_indices[:idx - self.n_groups_ - 1] == group)
                return lmbda * alpha[idx_in_group]
        else:
            return lmbda if idx == 0 else lmbda * alpha[idx - 1]

    def _make_key_2_hyperparams_dict(self):
        """
        Create dictionary mapping optimization keys to hyperparameters.
        """
        if self.lmbda1 is None:
            lmbda1 = 2. * np.sqrt(self.group_sizes_ * self.n_nodes_)
        else:
            lmbda1 = if_scalar_or_given_length_array(self.lmbda1, length=self.n_groups_, name="lmbda1")
        lmbda2 = 2. * self.n_layers_ * np.sqrt(self.n_nodes_ / self.n_groups_) if self.lmbda2 is None else self.lmbda2
        if self.alpha1 is None:
            alpha1 = np.ones(self.n_layers_) / np.sqrt(self.group_sizes_[self.group_indices])
        else:
            alpha1 = if_scalar_or_given_length_array(self.alpha1, length=self.n_layers_, name="alpha1")
        if self.alpha2 is None:
            alpha2 = self.group_sizes_ / self.n_layers_ * np.sqrt(self.n_groups_)
        else:
            alpha2 = if_scalar_or_given_length_array(self.alpha2, length=self.n_groups_, name="alpha2")

        self._key_2_hyperparams_dict = \
            {**{(0, group): {"lmbda1": lmbda1[group], "alpha1": alpha1[self.group_indices == group]}
                for group in self.unique_groups_},
             **{(1, 0): {"lmbda2": lmbda2, "alpha2": alpha2}}}

    def fit(self, As: List[np.array],
            lr: Union[float, Tuple[float, float]] = 1e-1,
            max_iter: int = 100, tol=1e-5, patience=10,
            refit=True, verbose: bool = False, verbose_interval=1,
            save_latent_history=False, save_nll_history=True):
        """
        Fit the GroupMultiNeSS model.

        Parameters
        ----------
        As : np.ndarray
            Input adjacency matrices.
        lr : float or tuple, default=0.1
            Learning rate(s) for stages.
        max_iter : int, default=100
            Maximum number of iterations.
        tol : float, default=1e-5
            Convergence tolerance.
        patience : int, default=10
            Early stopping patience.
        refit : bool, default=True
            Whether to refit components.
        verbose : bool, default=False
            Verbosity flag.
        verbose_interval : int, default=1
            Interval for printing verbose messages.
        save_latent_history : bool, default=False
            Whether to save latent history.
        save_nll_history : bool, default=True
            Whether to save NLL history.

        Returns
        -------
        self
            Fitted model instance.
        """
        As = self._validate_input(As=As)
        self._fit(As, lr=lr,
                  max_iter=max_iter, tol=tol, patience=patience,
                  refit=refit,
                  verbose=verbose, verbose_interval=verbose_interval,
                  save_latent_history=save_latent_history, save_nll_history=save_nll_history)
        return self


class GroupMultiNeSSCV(GroupMultiNeSS, BaseMultiNeSSRefinedCV):
    def __init__(self, group_indices: List[int],
                 param_grid: Dict[str, List] = None,
                 edge_distrib: str = "normal",
                 lmbda1: Union[float, np.ndarray, list] = None,
                 lmbda2: float = None,
                 alpha1: Union[float, np.ndarray, list] = None,
                 alpha2: Union[float, np.ndarray, list] = None,
                 loops_allowed: bool = True,
                 cv_folds=3, test_prop=0.1,
                 max_rank: int = None, init_rank: int = None,
                 sigmas: Union[float, List, np.ndarray] = None,
                 n_jobs: int = None,
                 refit_threshold=1e-8,
                 fix_layer_split: bool = False,
                 use_1se_rule: bool = False,
                 random_seed: int = None):

        BaseMultiNeSSRefinedCV.__init__(self, param_grid=param_grid, cv_folds=cv_folds, test_prop=test_prop,
                                        loops_allowed=loops_allowed,
                                        fix_layer_split=fix_layer_split,
                                        use_1se_rule=use_1se_rule,
                                        random_seed=random_seed)

        GroupMultiNeSS.__init__(self, group_indices=group_indices,
                                lmbda1=lmbda1, lmbda2=lmbda2, alpha1=alpha1, alpha2=alpha2, edge_distrib=edge_distrib,
                                max_rank=max_rank, init_rank=init_rank, sigmas=sigmas,
                                loops_allowed=loops_allowed, n_jobs=n_jobs,
                                refit_threshold=refit_threshold)

    def _make_key_2_hyperparams_grid_dict(self) -> None:
        const_range = np.array([0.03, 0.1, 0.3, 1, 3, 10])
        if self.param_grid is None:
            self._key_2_hyperparams_grid = {
                **{(1, 0): {"lmbda2": self.n_layers_ * np.sqrt(self.n_nodes_ / self.n_groups_) * const_range}},
                **{(0, group): {"lmbda1": np.sqrt(self.n_nodes_ * self.group_sizes_[group]) * const_range}
                   for group in self.unique_groups_}}
        else:

            stage_2_grid = {stage: {key: grid for key, grid in self.param_grid.items() if key.endswith(str(stage + 1))}
                            for stage in range(2)}
            self._key_2_hyperparams_grid = {**{(0, group): stage_2_grid[0] for group in self.unique_groups_},
                                            **{(1, 0): stage_2_grid[1]}}


class OracleGroupMultiNeSS(BaseGroupMultiNeSSRefined, BaseOracleMultiNeSS):
    def __init__(self, group_indices: List[int],
                 d_shared: int, d_groups: List[int], d_individs: List[int],
                 edge_distrib: str = "normal",
                 sigmas: Union[float, List, np.ndarray] = None,
                 loops_allowed: bool = True,
                 n_jobs: int = None):

        BaseGroupMultiNeSSRefined.__init__(self, group_indices=group_indices,
                                           edge_distrib=edge_distrib, sigmas=sigmas,
                                           loops_allowed=loops_allowed, n_jobs=n_jobs)
        self.d_shared = d_shared
        self.d_groups = d_groups
        self.d_individs = d_individs

    def _make_key_2_hyperparams_dict(self):
        d_shared = if_scalar_or_given_length_array(self.d_shared, length=1, name="d_shared")[0]
        d_groups = if_scalar_or_given_length_array(self.d_groups, length=self.n_groups_, name="d_groups")
        d_individs = if_scalar_or_given_length_array(self.d_individs, length=self.n_layers_, name="d_individs")
        self._key_2_hyperparams_dict = {**{(0, group): {"d_shared": d_shared + d_groups[group],
                                                        "d_individs": d_individs[self.group_indices == group]}
                                        for group in self.unique_groups_},
                                        **{(1, 0): {"d_shared": d_shared, "d_individs": d_groups}}}

    def _compute_threshold_rank(self, idx: int, key: Tuple[int, int]) -> int:
        hyperparam_dict = self._key_2_hyperparams_dict[key]
        d_shared, d_individs = hyperparam_dict["d_shared"], hyperparam_dict["d_individs"]
        stage, group = key
        if idx == 0:
            max_rank = d_shared
        elif 1 <= idx < self.n_groups_ + 1:
            if stage == 0:
                max_rank = d_shared
            else:
                max_rank = d_individs[idx - 1]
        else:
            idx_in_group = np.sum(self.group_indices[:idx - self.n_groups_ - 1] == group)
            max_rank = d_individs[idx_in_group]
        return max_rank
