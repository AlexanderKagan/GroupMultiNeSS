import numpy as np
from typing import Union, List
from warnings import warn

from .utils import sigmoid, generate_matrices_given_pairwise_max_cosines


class LatentPositionGenerator:
    def __init__(self, n_nodes, n_layers, *,
                 edge_distrib: str = "normal",
                 noise_sigma: float = 1.,
                 loops_allowed=True,
                 d_shared: int = 2,
                 d_individs: Union[List, int] = 2,
                 s_vu: float = 0.,
                 s_uu: float = 0.,
                 comps_max_cosine_mat: np.array = None,
                 min_V_max_U_eigval_ratio=None):

        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.edge_distrib = edge_distrib
        self.noise_sigma = noise_sigma
        self.loops_allowed = loops_allowed
        self.d_shared = d_shared
        self.d_individs = d_individs
        self.s_vu = s_vu
        self.s_uu = s_uu
        self.comps_max_cosine_mat = comps_max_cosine_mat
        self.min_V_max_U_eigval_ratio = min_V_max_U_eigval_ratio

    def _validate_input(self):
        if self.edge_distrib == "normal":
            self.link_fun = lambda x: x
        elif self.edge_distrib == "bernoulli":
            self.link_fun = sigmoid
        else:
            raise NotImplementedError("Link function should be normal or bernoulli")

        if isinstance(self.d_individs, (list, np.ndarray)):
            assert len(self.d_individs) == self.n_layers, \
                "d_individs should have the length equal to the layers number"
            self.d_individs_ = self.d_individs
        else:
            self.d_individs_ = self.d_individs * np.ones(self.n_layers, dtype=int)

        if self.comps_max_cosine_mat is not None:
            assert self.comps_max_cosine_mat.ndim == 2
            assert np.allclose(self.comps_max_cosine_mat, self.comps_max_cosine_mat.T), \
                "comps_max_cosine_mat matrix should be symmetric"
            assert np.all((self.comps_max_cosine_mat >= 0) & (self.comps_max_cosine_mat <= 1))
            assert np.allclose(np.all(np.diag(self.comps_max_cosine_mat)), 1)

    def _generate_latent_spaces(self):
        all_dims = [self.d_shared] + list(self.d_individs_)
        if np.sum(all_dims) <= self.n_nodes:
            if self.comps_max_cosine_mat is None:
                vu_block = self.s_vu * np.ones((1, self.n_layers))
                uu_block = self.s_uu * np.ones((self.n_layers, self.n_layers))
                self.comps_max_cosine_mat = np.block([[np.eye(1), vu_block],
                                                      [vu_block.T, uu_block]])
                np.fill_diagonal(self.comps_max_cosine_mat, 1)
            else:
                assert self.comps_max_cosine_mat.shape[0] == np.sum(all_dims)
            all_lat_spaces = generate_matrices_given_pairwise_max_cosines(self.n_nodes, ds=all_dims,
                                                                          pairwise_cos_mat=self.comps_max_cosine_mat)
        else:
            warn("When n_nodes < sum of all components' dimensions, all angle constraints cannot be satisfied.")
            all_lat_spaces = [np.random.randn(self.n_nodes, d) for d in all_dims]
        self.V, self.Us = all_lat_spaces[0], all_lat_spaces[1:]

        if self.min_V_max_U_eigval_ratio is not None:
            max_eigval_U = np.linalg.svd(self.V, compute_uv=False)[self.d_shared - 1] / self.min_V_max_U_eigval_ratio
            self.Us = [U / np.linalg.norm(U, ord=2) * max_eigval_U for U in self.Us]

    def _compute_shared_latent_position(self):
        self.S = self.V @ self.V.T

    def _compute_individual_latent_positions(self):
        self.Rs = np.stack([U @ U.T for U in self.Us])

    def _compute_latent_positions(self):
        self._compute_shared_latent_position()
        self._compute_individual_latent_positions()
        return self.S + self.Rs

    def generate(self, random_seed=None):
        self._validate_input()
        np.random.seed(random_seed)
        self._generate_latent_spaces()
        self.Ps = self.link_fun(self._compute_latent_positions())
        As = []

        triu_x, triu_y = np.triu_indices(self.n_nodes, k=1)
        for idx, P in enumerate(self.Ps):
            if self.edge_distrib == "normal":
                noise = self.noise_sigma * np.random.randn(self.n_nodes, self.n_nodes)
                noise[triu_x, triu_y] = noise[triu_y, triu_x]
                A = P + noise
            elif self.edge_distrib == "bernoulli":
                A = (np.random.rand(self.n_nodes, self.n_nodes) <= P).astype(float)
                A[triu_x, triu_y] = A[triu_y, triu_x]
            else:
                raise NotImplementedError()

            if not self.loops_allowed:
                np.fill_diagonal(A, 0.)
            As.append(A)
        self.As = np.stack(As)


class GroupLatentPositionGenerator(LatentPositionGenerator):
    def __init__(self, n_nodes: int, n_layers: int,  *,
                 group_indices: List[int],
                 edge_distrib: str = "normal",
                 noise_sigma: float = 1.,
                 loops_allowed: bool = True,
                 d_shared: int = 2,
                 d_individs: Union[List, int] = 2,
                 d_groups: Union[List, int] = 2,
                 s_vw: float = 0.,
                 s_vu: float = 0.,
                 s_ww: float = 0.,
                 s_wu: float = 0.,
                 s_uu: float = 0.,
                 comps_max_cosine_mat=None,
                 min_V_max_W_eigval_ratio=None):

        super().__init__(n_nodes, n_layers, edge_distrib=edge_distrib, noise_sigma=noise_sigma,
                         loops_allowed=loops_allowed, d_shared=d_shared, d_individs=d_individs,
                         comps_max_cosine_mat=comps_max_cosine_mat)

        self.group_indices = group_indices
        self.d_groups = d_groups
        self.unique_groups = np.sort(np.unique(group_indices))
        self.s_vw = s_vw
        self.s_vu = s_vu
        self.s_ww = s_ww
        self.s_wu = s_wu
        self.s_uu = s_uu
        self.min_V_max_W_eigval_ratio = min_V_max_W_eigval_ratio

    def _validate_input(self):
        super()._validate_input()
        assert (0 <= self.s_vw <= 1) & (0 <= self.s_uu <= 1) & (0 <= self.s_ww <= 1)
        assert len(self.group_indices) == self.n_layers, "Number of group indices should = the number of layers"
        assert np.all(np.sort(self.unique_groups) == np.arange(len(self.unique_groups))), \
            "group_indices should contain ints from 0 to n_groups - 1, where n_groups is the number of distinct groups"
        self.n_groups_ = len(self.unique_groups)
        assert self.n_groups_ >= 2, "Number of groups should be at least 2"
        if isinstance(self.d_groups, (list, np.ndarray)):
            assert self.n_groups_ == len(self.d_groups), \
                "Number of distinct groups should be the same as the length of d_groups"
            self.d_groups_ = self.d_groups
        else:
            self.d_groups_ = self.d_groups * np.ones(self.n_groups_, dtype=int)

    def _generate_latent_spaces(self):
        all_dims = [self.d_shared] + list(self.d_groups_) + list(self.d_individs_)
        if self.n_nodes >= np.sum(all_dims):
            if self.comps_max_cosine_mat is None:
                uu_block = self.s_uu * np.ones((self.n_layers, self.n_layers))
                ww_block = self.s_ww * np.ones((self.n_groups_, self.n_groups_))
                vw_block = self.s_vw * np.ones((1, self.n_groups_))
                vu_block = self.s_vu * np.ones((1, self.n_layers))
                wu_block = self.s_wu * np.ones((self.n_groups_, self.n_layers))
                self.comps_max_cosine_mat = np.block([[np.eye(1), vw_block, vu_block],
                                                      [vw_block.T, ww_block, wu_block],
                                                      [vu_block.T, wu_block.T, uu_block]])
                np.fill_diagonal(self.comps_max_cosine_mat, 1)
            else:
                assert self.comps_max_cosine_mat.shape[0] == np.sum(all_dims)

            all_lat_spaces = generate_matrices_given_pairwise_max_cosines(self.n_nodes, ds=all_dims,
                                                                          pairwise_cos_mat=self.comps_max_cosine_mat)
        elif self.n_nodes >= self.d_shared + np.sum(self.d_groups_):
            shared_group_dims = [self.d_shared] + list(self.d_groups_)
            ww_block = self.s_ww * np.ones((self.n_groups_, self.n_groups_))
            vw_block = self.s_vw * np.ones((1, self.n_groups_))
            cos_mat = np.block([[np.eye(1), vw_block],
                                [vw_block.T, ww_block]])
            np.fill_diagonal(cos_mat, 1)
            shared_group_lat_spaces = generate_matrices_given_pairwise_max_cosines(self.n_nodes, ds=shared_group_dims,
                                                                                   pairwise_cos_mat=cos_mat)
            all_lat_spaces = shared_group_lat_spaces + [np.random.randn(self.n_nodes, d) for d in self.d_individs_]
        else:
            warn("When n_nodes < sum of all components' dimensions, all angle constraints cannot be satisfied.")
            all_lat_spaces = [np.random.randn(self.n_nodes, d) for d in all_dims]

        self.V = all_lat_spaces[0]
        self.Ws = all_lat_spaces[1: self.n_groups_ + 1]
        self.Us = all_lat_spaces[self.n_groups_ + 1:]

        if self.min_V_max_W_eigval_ratio is not None:
            max_eigval_W = np.linalg.svd(self.V, compute_uv=False)[self.d_shared - 1] / self.min_V_max_W_eigval_ratio
            self.Ws = [W / np.linalg.norm(W, ord=2) * max_eigval_W for W in self.Ws]

    def _compute_group_latent_positions(self):
        self.Qs = np.stack([W @ W.T for W in self.Ws])

    def _compute_latent_positions(self):
        self._compute_shared_latent_position()
        self._compute_group_latent_positions()
        self._compute_individual_latent_positions()
        return self.S + self.Qs[self.group_indices] + self.Rs
