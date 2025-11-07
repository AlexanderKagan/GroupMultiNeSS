import numpy as np
from numpy.linalg import norm
from typing import List, Union
from warnings import warn
from typing import Iterable
from multiprocessing import shared_memory
import statsmodels.api as sm
from more_itertools import zip_equal

from .utils import sigmoid, leading_left_eigenvectors


class SharedMemoryMatrixFitter:
    """
    Base class for managing parameter matrices in shared memory.

    Allows storing, retrieving, and updating matrices in shared memory,
    which can be used for parallel computations without duplicating data.
    """

    def _init_param_matrices(self, vals: Union[List[np.ndarray], np.ndarray] = None,
                             shape: Iterable[int] = None) -> None:
        """
        Initialize shared memory for parameter matrices.

        Parameters
        ----------
        vals : list of np.ndarray or np.ndarray, optional
            Initial values for the matrices. If provided, shape is inferred.
        shape : iterable of int, optional
            Shape of the matrices to initialize if vals is not provided.

        Notes
        -----
        - If both vals and shape are provided, vals take precedence and shape is ignored.
        - Allocates shared memory of type float64 for storing matrices.
        """
        if shape is None:
            assert vals is not None, "One of vals or shape arguments should berandom_seed provided!"
            vals = np.array(vals)
            shape = vals.shape
        elif vals is not None:
            warn("shape argument is not used as vals argument is provided!")

        shm = shared_memory.SharedMemory(create=True, size=np.prod(shape) * 8)
        self._param_matrices_shm = shared_memory.SharedMemory(name=shm.name)
        self.param_matrices = np.ndarray(shape, dtype=np.float64, buffer=self._param_matrices_shm.buf)
        if vals is not None:
            self.param_matrices[:] = vals

    def _set_matrix(self, val, idx):
        """
        Set a specific matrix in the shared memory.

        Parameters
        ----------
        val : np.ndarray
            Matrix value to set.
        idx : int
            Index of the matrix to update.
        """
        param_matrices = self.get_all_fitted_matrices()
        param_matrices[idx] = val

    def _set_matrices(self, vals, indices: Union[List[int], np.ndarray] = None):
        """
        Set multiple matrices in shared memory.

        Parameters
        ----------
        vals : list[np.ndarray] or np.ndarray
            List of matrix values to set.
        indices : list or np.ndarray, optional
            Indices of matrices to update. If None, all matrices are updated.
        """
        if indices is None:
            indices = list(range(len(self.get_all_fitted_matrices())))
        assert len(vals) == len(indices)
        for idx, val in zip_equal(indices, vals):
            self._set_matrix(val, idx)

    def get_all_fitted_matrices(self):
        """
        Retrieve all matrices stored in shared memory.

        Returns
        -------
        np.ndarray
            Array backed by shared memory containing all parameter matrices.
        """
        return np.ndarray(self.param_matrices.shape, dtype=np.float64, buffer=self._param_matrices_shm.buf)


class BaseMultiplexNetworksModel:
    """
    Base class for multiplex network models.

    Handles edge distribution, loop constraints, and basic input validation.
    """

    def __init__(self, edge_distrib: str = "normal", loops_allowed: bool = True):
        """
        Initialize a multiplex network model.

        Parameters
        ----------
        edge_distrib : str, default='normal'
            Type of edge distribution ('normal' or 'bernoulli').
        loops_allowed : bool, default=True
            Whether self-loops are allowed in adjacency matrices.
        """
        self.edge_distrib = edge_distrib
        self.loops_allowed = loops_allowed
        if edge_distrib == "normal":
            self.link_ = lambda x: x
        elif edge_distrib == "bernoulli":
            self.link_ = sigmoid
        else:
            raise NotImplementedError("Edge distribution should be either normal or bernoulli!")

    def _validate_input(self, As: List[np.array]) -> np.ndarray:
        """
        Validate input adjacency matrices.

        Checks for:
        - Square matrices
        - Consistent shape across layers
        - Absence of loops if loops_allowed is False

        Parameters
        ----------
        As : list of np.ndarray
            List of adjacency matrices for each layer.

        Returns
        -------
        np.ndarray
            Stack of adjacency matrices as a 3D array.
        """
        assert As[0].shape[0] == As[0].shape[1], "adjacency matrix should have a square form"
        assert np.all([As[0].shape == A.shape for A in As[1:]]), "networks should share the same vertex set"
        if not self.loops_allowed:
            assert np.all(np.hstack([np.diag(A) for A in As]) == 0), \
                "loops present in one of adjacency matrices while loops_allowed == False"
        self.n_nodes_ = As[0].shape[0]
        self.n_layers_ = len(As)
        self._param_participance_masks = self._get_param_participance_masks()
        return np.stack(As)

    def _get_param_participance_masks(self) -> np.ndarray:
        """
        Return masks indicating which parameters participate in each layer.

        Must be implemented by subclasses.

        Returns
        -------
        np.ndarray
            Boolean mask array for parameter participation.
        """
        pass


class BaseRefitting(BaseMultiplexNetworksModel, SharedMemoryMatrixFitter):
    """
    Base class for refitting latent matrices in multiplex network models.
    """

    def __init__(self, edge_distrib="normal", max_rank: int = None, loops_allowed: bool = True,
                 refit_threshold: float = 1e-8):
        """
        Initialize the BaseRefitting object.

        Parameters
        ----------
        edge_distrib : str, default="normal"
            Type of edge distribution ('normal' or 'bernoulli').
        max_rank : int, optional
            Maximum rank for eigenvector-based refitting.
        loops_allowed : bool, default=True
            Whether self-loops are allowed in adjacency matrices.
        refit_threshold : float, default=1e-8
            Threshold to determine which matrices should be refitted.
        """
        BaseMultiplexNetworksModel.__init__(self, edge_distrib=edge_distrib, loops_allowed=loops_allowed)
        self.max_rank = max_rank
        self.refit_threshold = refit_threshold

    @property
    def n_fitted_matrices(self):
        """
        Number of matrices currently fitted in shared memory.

        Returns
        -------
        int
            Number of fitted matrices.
        """
        return len(self.get_all_fitted_matrices())

    def _compute_offset(self, refit_matrix_indices) -> np.ndarray:
        """
        Compute offset vector for refitting.

        Parameters
        ----------
        refit_matrix_indices : array-like
            Indices of matrices to refit.

        Returns
        -------
        np.ndarray
            Offset vector for GLM refitting.
        """
        offsets_over_obs = []
        obs_mask = self._param_participance_masks[refit_matrix_indices].any(0)
        triu_indices = np.triu_indices(self.n_nodes_, k=0 if self.loops_allowed else 1)
        for layer_particip_mask in self._param_participance_masks[:, obs_mask].T:
            offset_indices = np.setdiff1d(np.arange(self.n_fitted_matrices)[layer_particip_mask], refit_matrix_indices)
            obs_offset = self.get_all_fitted_matrices()[offset_indices].sum(0)[triu_indices]
            offsets_over_obs.append(obs_offset)
        return np.hstack(offsets_over_obs)

    def _construct_refit_design_response_and_offset(self, As: np.ndarray, refit_matrix_indices: np.ndarray,
                                                    refit_matrix_eigvecs: List[np.ndarray]):
        """
        Construct the design matrix, response vector, and offset for GLM refitting.

        Parameters
        ----------
        As : np.ndarray
            Stacked adjacency matrices.
        refit_matrix_indices : np.ndarray
            Indices of matrices to refit.
        refit_matrix_eigvecs : list of np.ndarray
            Eigenvectors for the refit matrices.

        Returns
        -------
        tuple of np.ndarray
            Design matrix, response vector, and offset vector for GLM fitting.
        """
        triu_indices = np.triu_indices(self.n_nodes_, k=0 if self.loops_allowed else 1)
        obs_mask = self._param_participance_masks[refit_matrix_indices].any(0)
        response = np.hstack([A[triu_indices] for A in As[obs_mask]])

        param_designs = [np.stack([np.outer(evec, evec)[triu_indices] for evec in eigvecs.T], axis=-1)
                         for eigvecs in refit_matrix_eigvecs]
        offset = self._compute_offset(refit_matrix_indices)
        refit_matrix_participance_masks = self._param_participance_masks[refit_matrix_indices][:, obs_mask]

        design_mat = []
        for param_design, participance_mask in zip_equal(param_designs, refit_matrix_participance_masks):
            design_mat.append(np.vstack([param_design if participate else np.zeros_like(param_design)
                                         for participate in participance_mask]))

        present_vals_mask = ~np.isnan(response)
        return np.hstack(design_mat)[present_vals_mask], response[present_vals_mask], offset[present_vals_mask]

    @staticmethod
    def _construct_refitted_matrices(refit_matrix_eigvecs: List[np.ndarray], refit_matrix_eigvals: np.ndarray):
        """
        Reconstruct matrices from eigenvectors and refitted eigenvalues.

        Parameters
        ----------
        refit_matrix_eigvecs : list of np.ndarray
            Eigenvectors for each refit matrix.
        refit_matrix_eigvals : np.ndarray
            Refitted eigenvalues from GLM.

        Returns
        -------
        list of np.ndarray
            Refitted matrices.
        """
        assert np.sum([mat.shape[1] for mat in refit_matrix_eigvecs]) == len(refit_matrix_eigvals)
        refitted_matrices = []
        start_idx = 0
        for matrix_eigvec in refit_matrix_eigvecs:
            matrix_dim = matrix_eigvec.shape[1]
            matrix_eigvals = refit_matrix_eigvals[start_idx: start_idx + matrix_dim]
            refit_matrix = matrix_eigvec @ np.diag(matrix_eigvals) @ matrix_eigvec.T
            refitted_matrices.append(refit_matrix)
            start_idx += matrix_dim
        return refitted_matrices

    def _preprocess_refit_matrix_indices(self, refit_matrix_indices=None) -> np.ndarray:
        """
        Preprocess the indices of matrices to refit by filtering out near-zero matrices.

        Parameters
        ----------
        refit_matrix_indices : iterable of int, optional
            Indices of matrices to consider for refitting.

        Returns
        -------
        np.ndarray
            Filtered array of matrix indices to refit.
        """
        if refit_matrix_indices is None:
            refit_matrix_indices = range(self.n_fitted_matrices)
        refit_matrix_indices = list(filter(lambda idx:
                                           norm(self.get_all_fitted_matrices()[idx], ord=2) > self.refit_threshold,
                                           refit_matrix_indices))  # don't refit rank zero matrices after hard-threshold
        return np.array(refit_matrix_indices, dtype=int)

    def refit(self, As: Union[List[np.array], np.array], refit_matrix_indices=None):
        """
        Refit parameter matrices using GLM based on current adjacency matrices.

        Parameters
        ----------
        As : list or np.ndarray
            Input adjacency matrices.
        refit_matrix_indices : iterable of int, optional
            Indices of matrices to refit. If None, all non-zero matrices are refitted.
        """
        As = self._validate_input(As)
        refit_matrix_indices = self._preprocess_refit_matrix_indices(refit_matrix_indices)
        if len(refit_matrix_indices) != 0:

            refit_matrix_eigvecs = [leading_left_eigenvectors(mat, k=self.max_rank,
                                                              eigval_threshold=self.refit_threshold)
                                    for mat in self.get_all_fitted_matrices()[refit_matrix_indices]]

            design_mat, response, offset = self._construct_refit_design_response_and_offset(
                As=As, refit_matrix_indices=refit_matrix_indices, refit_matrix_eigvecs=refit_matrix_eigvecs)

            family = sm.families.Gaussian() if self.edge_distrib == "normal" else sm.families.Binomial()
            refit_model = sm.GLM(response, design_mat, family=family, offset=offset)
            result = refit_model.fit()
            refit_matrix_eigvals = result.params

            refitted_matrices = self._construct_refitted_matrices(refit_matrix_eigvecs, refit_matrix_eigvals)
            self._set_matrices(refitted_matrices, indices=refit_matrix_indices)
