import numpy as np
from typing import List, Union, Tuple
from warnings import warn

from .utils import if_scalar_or_given_length_array, hard_thresholding_operator, psd_projection, truncated_svd
from .multiness import BaseMultiNeSS, OracleMultiNeSS


class SharedSpaceHunt(BaseMultiNeSS):
    """
    Estimate shared and individual latent spaces in multiplex networks using
    Shared Space Hunt method proposed in [1].

    References
    ----------
    [1] Tian, Y. et al. (2024). Efficient Analysis of Latent Spaces in Heterogeneous Networks. arXiv:2412.02151.
    """

    def __init__(self, d_shared: int = None, d_individs: Union[int, List[int]] = None, tau: float = None,
                 edge_distrib: str = "normal", loops_allowed: bool = True):
        """
        Initialize the SharedSpaceHunt object.

        Parameters
        ----------
        d_shared : int, optional
            Number of shared latent dimensions. If None, estimated automatically.
        d_individs : int or list of int, optional
            Number of individual latent dimensions per layer. If None, estimated automatically.
        tau : float, optional
            Threshold parameter for SVD truncation. Defaults to max(sqrt(2*log(n)), 2) if None.
        edge_distrib : str, default="normal"
            Edge distribution ("normal" or "bernoulli").
        loops_allowed : bool, default=True
            Whether self-loops are allowed in adjacency matrices.
        """
        super().__init__(edge_distrib=edge_distrib, loops_allowed=loops_allowed)
        self.d_shared = d_shared
        self.d_individs = d_individs
        self.tau = tau
        if edge_distrib == "normal":
            self.inv_link_ = lambda x: x
        elif edge_distrib == "bernoulli":
            self.inv_link_ = lambda x: np.log(x / (1. - x))

    def _validate_hyperparams(self):
        """
        Validate or set default hyperparameters (d_shared, d_individs, tau).
        """
        if self.tau is None:
            self.tau = max(np.sqrt(2 * np.log(self.n_nodes_)), 2)
        if self.d_individs is not None:
            self.d_individs = if_scalar_or_given_length_array(self.d_individs, length=self.n_layers_,
                                                              name="d_individs").astype(int)
            if self.d_shared is None:
                warn("d_individs would not be used in estimation if d_shared is not provided!")

    def _compute_latent_space_matrix(self, A: np.ndarray):
        """
        Compute a latent space matrix from a single adjacency matrix using hard thresholding
        and PSD projection.

        Parameters
        ----------
        A : np.ndarray
            Adjacency matrix.

        Returns
        -------
        np.ndarray
            Estimated latent space matrix.
        """
        eigval_threshold = np.linalg.norm(A, ord="fro") / np.sqrt(self.n_nodes_)
        Theta = hard_thresholding_operator(A, threshold=eigval_threshold)
        Theta = self.inv_link_(np.clip(Theta, self.link_(-5), self.link_(5)))
        return psd_projection(Theta)

    def _compute_latent_space_matrix_rank(self, Theta: np.ndarray):
        """
        Estimate the rank of a latent space matrix using a thresholding scheme.

        Parameters
        ----------
        Theta : np.ndarray
            Latent space matrix.

        Returns
        -------
        int
            Estimated rank.
        """
        thresholds = self.n_nodes_ ** (-1. / (4 * np.arange(1, self.n_nodes_) + 8))
        eigvals = np.linalg.svd(Theta, compute_uv=False)
        rank = 1 + np.argmax(eigvals[1:] <= eigvals[:-1] * thresholds)
        return rank

    def _compute_latent_space_positions_and_ranks(self, As: np.ndarray) -> Tuple[List, List]:
        """
        Compute latent space positions and ranks for all layers.

        Parameters
        ----------
        As : np.ndarray
            Stack of adjacency matrices.

        Returns
        -------
        Tuple[List, List]
            List of latent positions per layer and list of ranks per layer.
        """
        Ys = []
        ranks = []
        for t, A in enumerate(As):
            if self.d_shared is None or self.d_individs is None:
                Theta = self._compute_latent_space_matrix(A)
                rank = self._compute_latent_space_matrix_rank(Theta)
            else:
                Theta = A
                rank = self.d_shared + self.d_individs[t]
            u, s, _ = truncated_svd(Theta, max_rank=rank)
            Y = u @ np.diag(np.sqrt(s))
            Ys.append(Y)
            ranks.append(rank)
        return Ys, ranks

    def _estimate_shared_and_individ_latent_space_ranks(self, Ys, joint_ranks):
        """
        Estimate ranks for shared and individual latent spaces.

        Parameters
        ----------
        Ys : list of np.ndarray
            Latent positions per layer.
        joint_ranks : list of int
            Joint ranks per layer.
        """
        if self.d_shared is not None:
            self.d_shared_ = self.d_shared
        else:
            threshold = self.n_nodes_ ** (-1. / 8)
            K_ts = []
            for t in range(self.n_layers_):
                for s in range(t + 1, self.n_layers_):
                    Y_ts = np.hstack([Ys[t], Ys[s]])
                    sing_vals = np.linalg.svd(Y_ts, compute_uv=False)
                    null_rank = 1 + np.argmax(sing_vals[1:] <= sing_vals[0] * threshold)
                    K_ts.append(joint_ranks[t] + joint_ranks[s] - null_rank)
            self.d_shared_ = min(K_ts)
        if self.d_individs is None:
            self.d_individs_ = np.array(joint_ranks).astype(int) - self.d_shared_
        else:
            self.d_individs_ = self.d_individs

    def get_all_matrix_latent_positions(self, ds: List[int] = None, check_if_symmetric=True):
        """
        Return latent positions for shared and individual matrices.

        Parameters
        ----------
        ds : list of int, optional
            Dimension specifications (ignored here).

        check_if_symmetric : bool, default=True
            Check if matrices are symmetric.

        Returns
        -------
        list of np.ndarray
            Latent positions for all matrices.
        """
        ds = [self.d_shared_, self.d_individs_]
        return super().get_all_matrix_latent_positions(ds=ds)

    def _estimate_shared_latent_space_matrix(self, Ys):
        """
        Estimate the shared latent space component from latent positions Ys.

        Parameters
        ----------
        Ys : list of np.ndarray
            Latent positions per layer.
        """
        self.ts_pairs_ = []
        shared_comp = np.zeros((self.n_nodes_, self.n_nodes_))
        for t in range(self.n_layers_):
            for s in range(t + 1, self.n_layers_):
                Y_ts = np.hstack([Ys[t], Ys[s]])
                Y_tms = np.hstack([Ys[t], -Ys[s]])
                d_ts = self.d_shared_ + self.d_individs_[t] + self.d_individs_[s]
                _, s_vals, vh = np.linalg.svd(Y_ts)
                V_ts = vh.T[:, -self.d_shared_:]
                if s_vals[0] <= self.tau * s_vals[d_ts - 1]:
                    self.ts_pairs_.append((t, s))
                    shared_comp += Y_tms @ V_ts @ V_ts.T @ Y_tms.T
        if len(self.ts_pairs_) == 0:
            warn("Shared latent space component estimated as zero! Try increasing tau.")
        else:
            shared_comp = shared_comp / (2. * len(self.ts_pairs_))
        self._set_matrix(shared_comp, 0)

    def _estimate_individ_latent_space_matrices(self, Ys):
        """
        Estimate individual latent space components by removing shared component.

        Parameters
        ----------
        Ys : list of np.ndarray
            Latent positions per layer.
        """
        shared_comp = self.get_shared_latent_space()
        individ_comps = [Y @ Y.T - shared_comp for Y in Ys]
        individ_comp_indices = list(range(1, self.n_layers_ + 1))
        self._set_matrices(vals=individ_comps, indices=individ_comp_indices)

    def _pre_fit(self, As: np.ndarray):
        """
        Preprocessing before fitting: initialize latent matrices.

        Parameters
        ----------
        As : np.ndarray
            Stack of adjacency matrices.
        """
        M, n, _ = As.shape
        self._validate_hyperparams()
        self._init_param_matrices(shape=(M + 1, n, n))

    def _fit(self, As: np.ndarray):
        """
        Fit shared and individual latent space matrices.

        Parameters
        ----------
        As : np.ndarray
            Stack of adjacency matrices.
        """
        self._pre_fit(As)
        Ys, joint_ranks = self._compute_latent_space_positions_and_ranks(As)
        self._estimate_shared_and_individ_latent_space_ranks(Ys, joint_ranks)
        self._estimate_shared_latent_space_matrix(Ys)
        self._estimate_individ_latent_space_matrices(Ys)

    def fit(self, As: List[np.ndarray]):
        """
        Public fit method.

        Parameters
        ----------
        As : list of np.ndarray
            List of adjacency matrices.

        Returns
        -------
        self
        """
        As = self._validate_input(As)
        self._fit(As)
        return self


class SharedSpaceHuntRefined(OracleMultiNeSS):
    """
    SharedSpaceHunt with Refinement proposed in [1].

    References
    ----------
    [1] Tian, Y. et al. (2024). Efficient Analysis of Latent Spaces in Heterogeneous Networks. arXiv:2412.02151.
    """

    def __init__(self, tau: float = None, d_shared: int = None, d_individs: List[int] = None,
                 edge_distrib: str = "normal", loops_allowed: bool = True):
        """
        Initialize SharedSpaceHuntRefined.

        Parameters
        ----------
        tau : float, optional
            Threshold parameter for SVD.
        d_shared : int, optional
            Shared latent space dimension.
        d_individs : list of int, optional
            Individual latent space dimensions per layer.
        edge_distrib : str, default="normal"
            Edge distribution.
        loops_allowed : bool, default=True
            Whether self-loops are allowed.
        """
        super().__init__(d_shared=d_shared, d_individs=d_individs,
                         edge_distrib=edge_distrib, loops_allowed=loops_allowed)
        self.tau = tau
        self.d_shared = d_shared
        self.d_individs = d_individs

    def _make_key_2_hyperparams_dict(self):
        """Placeholder for hyperparameter dictionary creation."""
        pass

    def _optimized_params_init(self, As, key: Tuple[int, int]) -> None:
        """
        Initialize parameters using SharedSpaceHunt.

        Parameters
        ----------
        As : list of np.ndarray
            Adjacency matrices.
        key : tuple
            Key for parameter dictionary.
        """
        ssh = SharedSpaceHunt(d_shared=self.d_shared, d_individs=self.d_individs, tau=self.tau,
                              edge_distrib=self.edge_distrib, loops_allowed=self.loops_allowed)
        ssh.fit(As)
        self.d_shared_ = ssh.d_shared_
        self.d_individs_ = ssh.d_individs_
        self._key_2_hyperparams_dict = {(0, 0): {"d_shared": self.d_shared_, "d_individs": self.d_individs_}}
        self._set_matrices(ssh.get_all_fitted_matrices())
