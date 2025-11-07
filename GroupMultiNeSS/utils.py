import numpy as np
from scipy.sparse.linalg import svds
from typing import List, Union, Tuple
from copy import copy
from scipy.sparse.linalg import eigsh
from sklearn.model_selection import KFold


class MultipleNetworkTrainTestSplitter:
    """
    Class to generate train/test splits for multiple adjacency matrices
    representing layers of a multiplex network. Supports k-fold cross-validation
    or a simple random split. Can handle symmetric adjacency matrices and optionally
    ignores diagonal entries when loops are not allowed.

    Parameters
    ----------
    loops_allowed : bool, default=True
        Whether self-loops (diagonal entries) are allowed in the adjacency matrices.
    fix_layer_split : bool, default=True
        Whether to use the same train/test mask for all layers.
    random_seed : int or None, default=None
        Seed for reproducibility.
    check_if_sym_input : bool, default=False
        Whether to assert that input adjacency matrices are symmetric.

    Methods
    -------
    train_test_split(As, test_prop=0.1)
        Returns train/test split of the adjacency matrices with given proportion.
    kfold_split(As, n_splits=5)
        Yields train/test splits according to k-fold cross-validation.
    """
    def __init__(self, loops_allowed=True, fix_layer_split=True, random_seed=None, check_if_sym_input=False):
        self.loops_allowed = loops_allowed
        self.fix_layer_split = fix_layer_split
        self.random_seed = random_seed
        self.check_if_sym_input = check_if_sym_input

    def _validate_input(self, As: np.array):
        assert As.ndim == 3
        assert As.shape[1] == As.shape[2]
        if self.check_if_sym_input:
            assert np.all([np.allclose(A, A.T) for A in As])
        self.n_layers_, self.n_nodes_, _ = As.shape

    def _make_train_test_layers(self, As, triu_test_masks: np.array):
        triu_x, triu_y = np.triu_indices(self.n_nodes_, k=0 if self.loops_allowed else 1)
        triu_train_masks = ~triu_test_masks
        As_test = As.copy()
        As_train = As.copy()
        for i, (test_mask, train_mask) in enumerate(zip(triu_test_masks, triu_train_masks)):
            test_triu_x, test_triu_y = triu_x[test_mask], triu_y[test_mask]
            train_triu_x, train_triu_y = triu_x[train_mask], triu_y[train_mask]
            As_train[i][test_triu_x, test_triu_y] = np.nan
            As_train[i][test_triu_y, test_triu_x] = np.nan
            As_test[i][train_triu_x, train_triu_y] = np.nan
            As_test[i][train_triu_y, train_triu_x] = np.nan
        return As_train, As_test

    def train_test_split(self, As, test_prop: float = 0.1):
        self._validate_input(As=As)
        triu_x, triu_y = np.triu_indices(self.n_nodes_, k=0 if self.loops_allowed else 1)
        np.random.seed(self.random_seed)
        if self.fix_layer_split:
            triu_test_masks = np.stack([np.random.rand(len(triu_x)) <= test_prop] * self.n_layers_)
        else:
            triu_test_masks = np.random.rand(self.n_layers_, len(triu_x)) <= test_prop
        return self._make_train_test_layers(As=As, triu_test_masks=triu_test_masks)

    def kfold_split(self, As, n_splits=5):
        self._validate_input(As=As)
        kfold = KFold(n_splits=n_splits, random_state=self.random_seed, shuffle=True)
        triu_x, triu_y = np.triu_indices(self.n_nodes_, k=0 if self.loops_allowed else 1)
        if self.fix_layer_split:
            rng = np.arange(len(triu_x))
            for _, test_idx in kfold.split(rng):
                triu_test_masks = np.stack([np.isin(rng, test_idx)] * self.n_layers_)
                yield self._make_train_test_layers(As, triu_test_masks)
        else:
            rng = np.arange(self.n_layers_ * len(triu_x))
            for _, test_idx in kfold.split(rng):
                triu_test_masks = np.isin(rng, test_idx).reshape((self.n_layers_, len(triu_x)))
                yield self._make_train_test_layers(As, triu_test_masks)


class EarlyStopper:
    def __init__(self, patience=10, higher_better=True, rtol=1e-4):
        self.patience = patience
        self.higher_better = higher_better
        self.rtol = rtol
        self.patience_counter = -1
        self.best_metric = None
        
    def stop_check(self, val) -> bool:
        if self.patience_counter == -1:
            self.best_metric = val
            self.patience_counter = 0
            return False
            
        tolerance = self.rtol * abs(self.best_metric)
        if self.higher_better:
            improvement = val > self.best_metric + tolerance
        else:
            improvement = val < self.best_metric - tolerance

        if improvement:
            self.patience_counter = 0
            self.best_metric = val
        else:
            self.patience_counter += 1
            
        return self.patience_counter >= self.patience


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def fill_nan(x: np.ndarray, val: float = 0.):
    return np.where(np.isnan(x), val, x)


def leading_left_eigenvectors(A, k: int, eigval_threshold=None):
    assert len(A.shape) == 2, "Matrix should have two dimensions"
    if k >= min(*A.shape):
        u, s, _ = np.linalg.svd(A, compute_uv=True)
    else:
        u, s, _ = svds(A, k=k)
        s_indices = np.argsort(s)[::-1]
        u, s = u[:, s_indices], s[s_indices]
    return u if eigval_threshold is None else u[:, s >= eigval_threshold]


def leading_eigenvectors(A, k: int):
    assert len(A.shape) == 2, "Matrix should have two dimensions"
    if k >= min(*A.shape):
        u, s, vt = np.linalg.svd(A, compute_uv=True)
    else:
        u, s, vt = svds(A, k=k)
        s_indices = np.argsort(s)[::-1]
        u, s, vt = u[:, s_indices], s[s_indices], vt[s_indices]
    return u, vt.T


def extract_single_member_from_each_group(objects, group_indices) -> list:
    unique_objects = []
    for group in sorted(np.unique(group_indices)):
        unique_objects.append(objects[group_indices == group][0])
    return unique_objects


def psd_projection(A):
    assert np.allclose(A, A.T), "A should be a symmetric square matrix"
    eigvals, eigvecs = np.linalg.eig(A)
    eigvals, eigvecs = np.real(eigvals), np.real(eigvecs)
    trunc_eigvals = np.clip(eigvals, 0, None)
    return eigvecs @ np.diag(trunc_eigvals) @ eigvecs.T


def truncated_eigen_decomposition(A, max_rank=None, which="LM"):
    assert np.allclose(A, A.T), "A should be a symmetric square matrix"
    if max_rank is None or max_rank >= len(A):
        eigvals, eigvecs = np.linalg.eig(A)
    else:
        assert which in ["LM", "LA"], "which should be either LM (largest magnitude) or LA (largest algebraic)"
        eigvals, eigvecs = eigsh(A, k=max_rank, which=which)
    return eigvals, np.real(eigvecs)


def truncated_svd(A, max_rank=None, compute_uv=True):
    min_dim = min(*A.shape)
    if max_rank is None or max_rank >= min_dim:
        max_rank = min_dim
    if max_rank == min_dim:
        u, s, vt = np.linalg.svd(A, compute_uv=compute_uv)
        u, vt = u[:, :min_dim], vt[:min_dim, :]
    elif max_rank == 0:
        u, s, vt = np.zeros((A.shape[0], 1)), np.array([0.]), np.zeros((1, A.shape[1]))
    else:
        u, s, vt = svds(A, k=max_rank, return_singular_vectors=compute_uv)
        s_indices = np.argsort(s)[::-1]
        u, s, vt = u[:, s_indices], s[s_indices], vt[s_indices]
    return u, s, vt


def hard_thresholding_operator(A, threshold=None, max_rank=None):
    u, s, vt = truncated_svd(A, max_rank=max_rank)
    if threshold is not None:
        s = np.where(s >= threshold, s, 0.)
    return u @ np.diag(s) @ vt


def soft_thresholding_operator(A, threshold, max_rank=None):
    u, s, vt = truncated_svd(A, max_rank=max_rank)
    s = np.clip(s - threshold, 0, None)
    return u @ np.diag(s) @ vt


def matrix_power(A: np.array, p: Union[int, float], eps=1e-8):
    if isinstance(p, int):
        return np.linalg.matrix_power(A, p)
    u, s, vt = np.linalg.svd(A)
    assert np.all(s > eps), "float powers defined only for matrices with all positive singular values"
    return u @ np.diag(s ** p) @ vt


def frobenius_error(A_true, A_pred, relative=False, include_offdiag=True):
    assert A_true.shape == A_pred.shape
    assert A_true.ndim <= 2
    mask = np.ones(A_true.shape, dtype=bool) if include_offdiag else ~np.eye(A_true.shape[0], dtype=bool)
    abs_error = np.sqrt(np.sum((A_true - A_pred) ** 2, where=mask))
    return abs_error / np.sqrt(np.sum(A_true ** 2,  where=mask)) if relative else abs_error


def mean_frobenius_error(As_true, As_pred, relative=True, include_offdiag=True):
    assert len(As_true) == len(As_pred)
    assert As_true.ndim in (2, 3)
    if As_true.ndim == 2:
        As_true = [As_true]
        As_pred = [As_pred]
    return np.mean([frobenius_error(A_true, A, relative=relative, include_offdiag=include_offdiag)
                    for A_true, A in zip(As_true, As_pred)])


def make_error_report(params_true: Union[List[np.array], Tuple[np.array]],
                      params_pred: Union[List[np.array], Tuple[np.array]],
                      relative_errors: bool = True):
    assert len(params_true) == len(params_pred)
    errs = []
    for true, pred in zip(params_true, params_pred):
        err = mean_frobenius_error(true, pred, relative=relative_errors)
        errs.append(err)
    return errs


def make_group_averages(As, group_indices, groups=None) -> np.array:
    A_bars = []
    if groups is not None:
        assert set(groups).issubset(set(group_indices))
    else:
        groups = sorted(np.unique(group_indices))
    for group in groups:
        group_mask = group_indices == group
        A_bar = np.nansum(As[group_mask], axis=0) / group_mask.sum()
        A_bars.append(A_bar)
    return np.stack(A_bars)


def random_orthonormal_matrix(m: int, n: int):
    assert m >= n
    random_matrix = np.random.randn(m, n)
    Q, R = np.linalg.qr(random_matrix)
    return Q[:, :n]


def generate_correlated_matrix(A, d, cor=0., max_eigval=None, eps=1e-5):
    n, d0 = A.shape
    assert d0 > 0
    proj_A = A @ np.linalg.pinv(A.T @ A + eps * np.eye(d0)) @ A.T
    A_cor = (np.eye(n) - proj_A) @ np.random.randn(n, d)    # uncorrelated matrix
    scales = np.sqrt(1. - cor ** 2) * np.ones(min(d, d0))
    Q = random_orthonormal_matrix(d0, min(d, d0)).T
    if d > d0:
        Q = np.vstack([Q, np.zeros((d - d0, d0))])
        scales = np.hstack([scales, np.ones(d - d0)])
    A_cor = cor * (A @ Q.T) + A_cor @ np.diag(scales)
    if max_eigval is not None:
        A_cor = A_cor / np.linalg.norm(A_cor, ord=2) * max_eigval
    return A_cor


def rectangular_eye(n: int, d: int):
    min_dim = min(n, d)
    eye = np.eye(min_dim)
    if n > d:
        eye = np.vstack([eye, np.zeros((n - d, d))])
    elif n < d:
        eye = np.hstack([eye, np.zeros((n, d - n))])
    return eye


def generate_matrices_given_pairwise_max_cosines(n: int, ds: list[int], pairwise_cos_mat: np.ndarray):
    assert pairwise_cos_mat.ndim == 2
    assert len(ds) == len(pairwise_cos_mat)
    assert np.allclose(pairwise_cos_mat, pairwise_cos_mat.T), "pairwise_cos_mat should be symmetric"
    assert n >= np.sum(ds)
    assert np.allclose(np.diag(pairwise_cos_mat), 1)
    assert np.all(pairwise_cos_mat >= 0) & np.all(pairwise_cos_mat <= 1), "pairwise_cos_mat entries should be in [0, 1]"
    m = len(ds)
    stacked_Vs = np.random.randn(n, np.sum(ds))
    cur_gram_mat = stacked_Vs.T @ stacked_Vs / n
    target_gram_mat = np.block([[pairwise_cos_mat[i, j] * rectangular_eye(ds[i], ds[j]) for j in range(m)]
                                for i in range(m)])
    stacked_Vs = stacked_Vs @ matrix_power(cur_gram_mat, -0.5) @ matrix_power(target_gram_mat, 0.5)
    cumsum_ds = [0] + list(np.cumsum(ds))
    return [stacked_Vs[:, s: e] for s, e in zip(cumsum_ds[:-1], cumsum_ds[1:])]


def make_group_indices(group_props, num_layers):
    groups_sizes = (np.array(group_props) * num_layers).astype(int)
    groups_sizes[-1] = num_layers - groups_sizes[:-1].sum()

    group_indices = np.hstack([[group_idx for _ in range(size)]
                              for group_idx, size in enumerate(groups_sizes)])
    return group_indices


def estimate_sigma_mad(M):
    # Center columns
    Mc = M - np.mean(M, axis=0)

    # Dimensions
    n, p = Mc.shape

    # Auxiliary quantities
    beta = min(n, p) / max(n, p)
    lambdastar = np.sqrt(
        2 * (beta + 1) + 8 * beta / (beta + 1 + np.sqrt(beta ** 2 + 14 * beta + 1))
    )
    wbstar = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43

    # Sigma estimate
    singular_values = np.linalg.svd(Mc, compute_uv=False)
    sigma = np.median(singular_values) / (np.sqrt(max(n, p)) * (lambdastar / wbstar))

    return sigma


def if_scalar_or_given_length_array(val, length: int, name: str = None) -> np.array:
    name = f'{val=}'.split('=')[0] if name is None else name
    if isinstance(val, (np.ndarray, list)):
        assert len(val) == length, \
            f"If {name} is not a scalar, it should have length {len(val)}"
        val = np.array(val)
    elif np.isscalar(val):
        val = val * np.ones(length)
    else:
        raise NotImplementedError(f"{name} should be a list, np.ndarray or scalar!")
    return val


def fill_diagonals(As: Union[np.array, List[np.array]], val: float = 0, inplace: bool = True):
    if not inplace:
        As = copy(As)
    for idx in range(len(As)):
        np.fill_diagonal(As[idx], val)
    if not inplace:
        return As


def pairwise_metric_matrix(matrices, metric, **metric_kwargs):
    dist_mat = np.zeros((len(matrices), len(matrices)))
    for i in range(len(matrices)):
        for j in range(i, len(matrices)):
            dist = metric(matrices[i], matrices[j], **metric_kwargs)
            dist_mat[i, j] = dist
            dist_mat[j, i] = dist
    return dist_mat


def cos_sim(A, B, max_rank=10):
    assert A.ndim == 2
    assert B.ndim == 2
    assert A.shape[0] == B.shape[0]
    U1, _, _ = truncated_svd(A, max_rank=max_rank)
    U2, _, _ = truncated_svd(B, max_rank=max_rank)
    return np.linalg.norm(U1.T @ U2, ord=2)


def avg_lp_error(y_true, y_pred, p=2):
    assert y_true.shape == y_true.shape
    return (np.nanmean((y_true - y_pred) ** p)) ** (1. / p)


def fisher_transform(x, eps=1e-6):
    assert np.all(x <= 1) and np.all(x >= -1)
    x = np.clip(x, -1 + eps, 1 - eps)
    return np.log((1. + x) / (1. - x)) / 2
