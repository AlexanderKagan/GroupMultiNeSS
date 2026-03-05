import numpy as np
from typing import List

from .utils import truncated_svd, truncated_eigen_decomposition


def MASE(As: np.array, d_shared: int, d_individs: List[int]):
    """
        Compute the Multiple Adjacency Spectral Embedding (MASE) of a set of adjacency matrices, method proposed in [1].

        Parameters
        ----------
        As : np.array
            List of adjacency matrices to embed.
        d_shared : int
            Dimension of the shared latent space across all layers.
        d_individs : List[int]
            List of dimensions for individual latent spaces of each adjacency matrix.

        Returns
        -------
        ps : List[np.ndarray]
            Reconstructed matrices projected onto the shared space.
        u_joint : np.ndarray
            Shared latent positions.
        rs : List[np.ndarray]
            Latent-space representations of each adjacency matrix.

        References
        ----------
        [1]  Arroyo, J., Athreya, A., Cape, J., Chen, G., Priebe, C. E., & Vogelstein, J. T. (2021).
        Inference for multiple heterogeneous networks with a common invariant subspace.
        Journal of Machine Learning Research, 22, 1‑49.
        """
    assert len(As) == len(d_individs)
    us = []
    for A, d_individ in zip(As, d_individs):
        _, eigenvectors = truncated_eigen_decomposition(A, max_rank=d_individ)
        us.append(eigenvectors)
    u_joint, _, _ = truncated_svd(np.hstack(us), max_rank=d_shared)
    rs = [u_joint.T @ A @ u_joint for A in As]
    ps = [u_joint @ r @ u_joint.T for r in rs]
    return ps, u_joint, rs


def ASE(A: np.array, d: int, check_if_symmetric: bool = True,
        indef_inner_prod: bool = True, return_signature=False):
    """
    Compute the Adjacency Spectral Embedding (ASE) of a symmetric adjacency matrix.

    Parameters
    ----------
    A : np.array
        Symmetric adjacency matrix to embed.
    d : int
        Embedding dimension.
    check_if_symmetric : bool, optional
        Whether to check symmetry of A. Default is True.
    indef_inner_prod: bool, optional
        If True, indefinite inner product with signature (p, d - p)  is assumed (GRDPG model).
        If False, standard inner product is assumed (RDPG model)
    return_signature: bool, optional
        Whether to return_signature of the inner product. Only used when indef_inner_prod is TrueDefault is False.

    Returns
    -------
    np.ndarray
        Latent position matrix of shape (n_nodes, d).

    References
    ----------
    Sussman, D. L., Tang, M., Fishkind, D. E., & Priebe, C. E. (2012). A consistent adjacency spectral embedding
    for stochastic blockmodel graphs and some of its applications. Journal of Computational and Graphical Statistics
    """
    if check_if_symmetric:
        assert np.allclose(A, A.T), "A should be a symmetric matrix"
    eigvals, eigvecs = truncated_eigen_decomposition(A, max_rank=d, which="LM" if indef_inner_prod else "LA")
    if indef_inner_prod:
        sort_idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sort_idx]
        eigvecs = eigvecs[:, :sort_idx]
        signature = np.where(eigvals > 0, 1., -1)
        ase = eigvecs @ np.diag(np.sqrt(np.abs(eigvals)))
        return (ase, signature) if return_signature else ase
    else:
        ase = eigvecs @ np.diag(np.sqrt(np.clip(eigvals, a_min=0, a_max=None)))
        return ase
