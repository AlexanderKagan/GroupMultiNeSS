import numpy as np
from typing import List, Optional, Union
from copy import deepcopy
from collections import Counter
from functools import partial
from joblib import Parallel, delayed
from warnings import warn
from contextlib import ExitStack, closing

from .multiness import BaseMultiNeSS, BaseMultiNeSSRefined, BaseMultiNeSSRefinedCV, BaseOracleMultiNeSS
from .base import BaseRefitting
from .utils import if_scalar_or_given_length_array, fill_diagonals, \
    soft_thresholding_operator, hard_thresholding_operator, frobenius_error


class BaseGroupMultiNeSS(BaseMultiNeSS):
    """
    Base class for group-structured MultiNeSS models.

    Handles shared, group-specific, and individual latent spaces for network layers.

    Parameters
    ----------
    group_indices : List[int]
        List of group assignments for each network layer.
    edge_distrib : str, default='normal'
        Distribution of edges ('normal', 'bernoulli', or 'poisson').
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
        self.group_indices = np.asarray(group_indices, dtype=int)

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

    def _validate_input(self, As: Union[List[np.ndarray], np.ndarray]):
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
                 max_rank: Optional[int] = None, init_rank: Optional[int] = None,
                 sigmas: Optional[Union[float, List, np.ndarray]] = None,
                 loops_allowed: bool = True,
                 n_jobs: Optional[int] = None):
        """
        Initialize BaseGroupMultiNeSSRefined.
        """
        BaseMultiNeSSRefined.__init__(self, edge_distrib=edge_distrib, loops_allowed=loops_allowed,
                                      max_rank=max_rank, init_rank=init_rank, sigmas=sigmas)
        self.group_indices = np.asarray(group_indices, dtype=int)
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
        return [*individs_within_group_indices, 1 + group]

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
        self._key_2_history_key = {0: 'Shared component', **{group + 1: f'GroupOptimize({group + 1})'
                                           for group in self.unique_groups_}}

    def _make_key_2_param_indices_dict(self):
        self._key_2_param_indices = {0: [0], **{group + 1: self._get_group_dependent_matrix_indices(group)
                                              for group in self.unique_groups_}}

    def _optimized_params_init(self, As, key=0):
        """Equation (13): truncate the group average before subtracting S."""
        As = self._initialization_data(As)
        shared = hard_thresholding_operator(np.nansum(As, axis=0) / self.n_layers_,
                                           max_rank=self.init_rank)
        self._set_matrix(shared, 0)
        group_rank = None if self.init_rank is None else min(2 * self.init_rank, self.n_nodes_)
        for group in self.unique_groups_:
            mask = self.group_indices == group
            group_average = hard_thresholding_operator(np.nansum(As[mask], axis=0) / mask.sum(),
                                                        max_rank=group_rank)
            self._set_matrix(group_average - shared, group + 1)
            for layer in np.flatnonzero(mask):
                residual = np.where(np.isnan(As[layer]), 0., As[layer] - group_average)
                self._set_matrix(hard_thresholding_operator(residual, max_rank=self.init_rank),
                                 self.n_groups_ + layer + 1)

    def _run_optimization(self, As, lr=None, n_outer_iter=10, **optim_kwargs):
        """Algorithm 1: independent group descents, then one shared update."""
        lr = self._learning_rate(lr)
        stable_iterations = 0
        for outer_iter in range(n_outer_iter):
            previous = self.get_shared_latent_space().copy()
            # Threads share parameter storage and histories; groups write disjoint
            # matrices and S stays fixed until all group descents have finished.
            Parallel(n_jobs=self.n_jobs, require='sharedmem')(
                delayed(self._optimize_params)(key=group + 1, As=As, lr=lr,
                                               outer_iter=outer_iter, **optim_kwargs)
                for group in self.unique_groups_)
            self._optimization_step([0], key=0, As=As, lr=lr, iteration=0,
                                    outer_iter=outer_iter, tol=optim_kwargs['tol'],
                                    verbose=optim_kwargs.get('verbose', False) and
                                    outer_iter % optim_kwargs.get('verbose_interval', 1) == 0,
                                    save_latent_history=optim_kwargs.get('save_latent_history', False),
                                    save_nll_history=optim_kwargs.get('save_nll_history', True))
            change = frobenius_error(previous, self.get_shared_latent_space(), relative=True)
            self.convergence_history_[outer_iter][0].append(change)
            stable_iterations = stable_iterations + 1 if change <= optim_kwargs['tol'] else 0
            self.n_outer_iter_ = outer_iter + 1
            if stable_iterations >= optim_kwargs['patience']:
                break

    def _error_components(self, key, true_params, estimates):
        if key == 0:
            return [(r'$S$', true_params[0], estimates[0])]
        group = key - 1
        mask = self.group_indices == group
        return [(rf'$Q_{{{key}}}$', np.asarray(true_params[1])[group], estimates[1][group]),
                (rf'$R_{{{key}\ell}}$', np.asarray(true_params[2])[mask], estimates[2][mask])]


class GroupMultiNeSS(BaseGroupMultiNeSSRefined, BaseRefitting):
    """GroupMultiNeSS with direct nuclear-norm penalties.

    rho_0 penalizes S, rho_k penalizes each Q_k, and rho_kl penalizes each R_kl.
    None defaults are 2*sqrt(M*n), 2*sqrt(m_k*n), and 2*sqrt(n), respectively.
    rho_k and rho_kl accept scalars or sequences in group and input-layer order.
    Group labels remain 0..K-1; optimization/history keys are 0 for S and 1..K
    for group descents. init_rank=None leaves initialization untruncated.
    """

    def __init__(self, group_indices, rho_0=None, rho_k=None, rho_kl=None,
                 edge_distrib='normal', max_rank=None, init_rank=None, sigmas=None,
                 loops_allowed=True, n_jobs=None, refit_threshold=1e-8):
        BaseGroupMultiNeSSRefined.__init__(self, group_indices=group_indices, edge_distrib=edge_distrib,
                                           max_rank=max_rank, init_rank=init_rank, sigmas=sigmas,
                                           loops_allowed=loops_allowed, n_jobs=n_jobs)
        BaseRefitting.__init__(self, edge_distrib=edge_distrib, max_rank=max_rank,
                               loops_allowed=loops_allowed, refit_threshold=refit_threshold)
        self.rho_0 = rho_0
        self.rho_k = rho_k
        self.rho_kl = rho_kl

    def _compute_projected_map(self, idx, lr, key):
        mask = self._param_participance_masks[idx]
        penalty = self._compute_penalty_coef(idx=idx, key=key)
        return partial(soft_thresholding_operator, threshold=penalty * lr / mask.sum(), max_rank=self.max_rank)

    def _compute_penalty_coef(self, idx, key):
        params = self._key_2_hyperparams_dict[key]
        if idx == 0:
            return params['rho_0']
        if idx == key:
            return params['rho_k']
        layer = idx - self.n_groups_ - 1
        idx_in_group = np.sum(self.group_indices[:layer] == key - 1)
        return params['rho_kl'][idx_in_group]

    def _make_key_2_hyperparams_dict(self):
        rho_0 = 2. * np.sqrt(self.n_layers_ * self.n_nodes_) if self.rho_0 is None else self.rho_0
        if not np.isscalar(rho_0):
            raise ValueError('rho_0 must be a scalar')
        rho_k = if_scalar_or_given_length_array(
            2. * np.sqrt(self.group_sizes_ * self.n_nodes_) if self.rho_k is None else self.rho_k,
            length=self.n_groups_, name='rho_k')
        rho_kl = if_scalar_or_given_length_array(
            2. * np.sqrt(self.n_nodes_) if self.rho_kl is None else self.rho_kl,
            length=self.n_layers_, name='rho_kl')
        self._validate_penalty(rho_0, 'rho_0')
        self._validate_penalty(rho_k, 'rho_k')
        self._validate_penalty(rho_kl, 'rho_kl')
        self._key_2_hyperparams_dict = {0: {'rho_0': rho_0}, **{
            group + 1: {'rho_k': rho_k[group], 'rho_kl': rho_kl[self.group_indices == group]}
            for group in self.unique_groups_}}

    def fit(self, As, lr=None, max_iter=100, tol=1e-3, patience=10, refit=True,
            n_outer_iter=10, verbose=False, verbose_interval=1,
            save_latent_history=False, save_nll_history=True):
        """Fit Algorithm 1, then jointly refit every nonzero parameter matrix.

        max_iter limits each group descent; n_outer_iter defaults to 10 outer iterations.
        tol and patience control consecutive small RFE changes in Q_k and S.
        lr is a common scalar scale, divided once by each matrix's layer
        participation count. Defaults are 1 for Gaussian/Poisson and 3 for Bernoulli.
        All distributions use fixed steps without backtracking.
        Each fit initializes all parameter matrices from the supplied data.
        """
        self._validate_optim_params(max_iter, tol, patience, verbose_interval, n_outer_iter)
        lr = self._learning_rate(lr)
        As = self._validate_input(As)
        self._fit(As, lr=lr, n_outer_iter=n_outer_iter, max_iter=max_iter, tol=tol, patience=patience,
                  refit=refit, verbose=verbose, verbose_interval=verbose_interval,
                  save_latent_history=save_latent_history, save_nll_history=save_nll_history)
        return self


class GroupMultiNeSSCV(GroupMultiNeSS, BaseMultiNeSSRefinedCV):
    """GroupMultiNeSS with shared or full first-outer-iteration edge CV.

    param_grid accepts rho_0 and, for full CV, rho_k. A rho_k grid can be a
    sequence used for every group, or a dict keyed by zero-based group labels.
    Values are actual penalties; rho_kl is linked to rho_k/sqrt(m_k).
    Default grids multiply each oracle rate by [0.03, 0.1, 0.3, 1, 3, 10].
    """

    def __init__(self, group_indices, param_grid=None, edge_distrib='normal',
                 rho_0=None, rho_k=None, rho_kl=None, loops_allowed=True,
                 cv_folds=3, test_prop=0.1, max_rank=None, init_rank=None,
                 sigmas=None, n_jobs=None, refit_threshold=1e-8,
                 fix_layer_split=False, use_1se_rule=False, random_seed=None, cv_type='shared'):
        BaseMultiNeSSRefinedCV.__init__(self, param_grid=param_grid, cv_folds=cv_folds,
                                        test_prop=test_prop, loops_allowed=loops_allowed,
                                        fix_layer_split=fix_layer_split, use_1se_rule=use_1se_rule,
                                        random_seed=random_seed, cv_type=cv_type)
        GroupMultiNeSS.__init__(self, group_indices=group_indices, rho_0=rho_0, rho_k=rho_k,
                                rho_kl=rho_kl, edge_distrib=edge_distrib, max_rank=max_rank,
                                init_rank=init_rank, sigmas=sigmas, loops_allowed=loops_allowed,
                                n_jobs=n_jobs, refit_threshold=refit_threshold)

    def _make_key_2_hyperparams_grid_dict(self):
        allowed = {'rho_0', 'rho_k'} if self.cv_type == 'full' else {'rho_0'}
        if self.param_grid is not None and set(self.param_grid) - allowed:
            raise ValueError(f'{self.cv_type} CV param_grid only accepts {sorted(allowed)}')
        BaseMultiNeSSRefinedCV._make_key_2_hyperparams_grid_dict(self)
        if self.cv_type == 'full':
            group_grid = None if self.param_grid is None else self.param_grid.get('rho_k')
            if isinstance(group_grid, dict) and set(group_grid) - set(self.unique_groups_):
                raise ValueError('rho_k grids must use existing zero-based group labels')
            self._key_2_hyperparams_grid.update({group + 1: {'rho_k': self._penalty_grid(
                'rho_k', np.sqrt(self.n_nodes_ * self.group_sizes_[group]), group=group)}
                for group in self.unique_groups_})

    def _set_cv_penalties(self, params, key):
        self._set_hyperparams(params, key)
        if key == 0 and self.cv_type == 'shared':
            rho_0 = params['rho_0']
            for group in self.unique_groups_:
                self._key_2_hyperparams_dict[group + 1]['rho_k'] = (
                    rho_0 * np.sqrt(self.group_sizes_[group] / self.n_layers_))
                self._key_2_hyperparams_dict[group + 1]['rho_kl'] = np.full(
                    self.group_sizes_[group], rho_0 / np.sqrt(self.n_layers_))
        elif key > 0:
            self._key_2_hyperparams_dict[key]['rho_kl'] = np.full(
                self.group_sizes_[key - 1], params['rho_k'] / np.sqrt(self.group_sizes_[key - 1]))

    def _cv_search(self, As, n_outer_iter, refit, optim_kwargs):
        if self.cv_type == 'full':
            self._cv_full(As, n_outer_iter, refit, optim_kwargs)
        else:
            self._cv_shared(As, n_outer_iter, refit, optim_kwargs)

    def _cv_full(self, As, n_outer_iter, refit, optim_kwargs):
        # Keep a fold's group fits with its own S initializer and edge mask. S CV
        # uses these same training/test edges, never full-data group estimates.
        folds = []
        lr = self._learning_rate(optim_kwargs['lr'])
        inner_kwargs = {**optim_kwargs, 'lr': lr}
        with ExitStack() as stack:
            for train, test in self._cv_splits(As):
                candidate = stack.enter_context(closing(self._new_cv_model(train, n_outer_iter)))
                initial = candidate.get_all_fitted_matrices().copy()
                folds.append((candidate, train, test, initial))
                for key in range(1, self.n_groups_ + 1):
                    mask = candidate._param_participance_masks[key]
                    for params in self._cv_results[key]:
                        candidate._set_matrices(initial)
                        candidate._set_cv_penalties({'rho_k': params[0]}, key)
                        candidate._optimize_params(key=key, As=train, **inner_kwargs)
                        self._cv_results[key][params].append(candidate._compute_nll(test, obs_mask=mask))
            for key in range(1, self.n_groups_ + 1):
                self._set_best_hyperparams(key)
            for candidate, train, test, initial in folds:
                candidate._set_matrices(initial)
                candidate._key_2_hyperparams_dict = deepcopy(self._key_2_hyperparams_dict)
                for key in range(1, self.n_groups_ + 1):
                    candidate._optimize_params(key=key, As=train, **inner_kwargs)
                group_fitted = candidate.get_all_fitted_matrices().copy()
                for params in self._cv_results[0]:
                    candidate._set_matrices(group_fitted)
                    candidate._set_cv_penalties({'rho_0': params[0]}, 0)
                    candidate._optimization_step([0], key=0, As=train, lr=lr, iteration=0,
                                                 outer_iter=0, verbose=False, tol=optim_kwargs['tol'],
                                                 save_latent_history=False, save_nll_history=False)
                    if refit:
                        candidate.refit(train)
                    self._cv_results[0][params].append(candidate._compute_nll(test))
            self._set_best_hyperparams(0)


class OracleGroupMultiNeSS(BaseGroupMultiNeSSRefined, BaseOracleMultiNeSS):
    def __init__(self, group_indices: List[int],
                 d_shared: int, d_groups: List[int], d_individs: List[int],
                 edge_distrib: str = "normal",
                 sigmas: Optional[Union[float, List[float], np.ndarray]] = None,
                 loops_allowed: bool = True,
                 n_jobs: Optional[int] = None):

        BaseGroupMultiNeSSRefined.__init__(self, group_indices=group_indices,
                                           edge_distrib=edge_distrib, sigmas=sigmas,
                                           loops_allowed=loops_allowed, n_jobs=n_jobs)
        self.d_shared = d_shared
        self.d_groups = d_groups
        self.d_individs = d_individs

    def _make_key_2_hyperparams_dict(self):
        d_shared = int(if_scalar_or_given_length_array(self.d_shared, length=1, name='d_shared')[0])
        d_groups = if_scalar_or_given_length_array(self.d_groups, length=self.n_groups_, name='d_groups')
        d_individs = if_scalar_or_given_length_array(self.d_individs, length=self.n_layers_, name='d_individs')
        self._key_2_hyperparams_dict = {0: {'d_shared': d_shared}, **{
            group + 1: {'d_group': int(d_groups[group]),
                        'd_individs': d_individs[self.group_indices == group]}
            for group in self.unique_groups_}}

    def _compute_threshold_rank(self, idx, key):
        params = self._key_2_hyperparams_dict[key]
        if idx == 0:
            return params['d_shared']
        if idx == key:
            return params['d_group']
        layer = idx - self.n_groups_ - 1
        idx_in_group = np.sum(self.group_indices[:layer] == key - 1)
        return int(params['d_individs'][idx_in_group])
