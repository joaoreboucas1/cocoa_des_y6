"""
Fisher-weighted principal components ("modes") of n(z) realizations.
Pure numpy; no cosmology code here.  Equation numbers refer to
Bernstein et al. 2025 (arXiv:2506.00758, v2).

Notation
--------
n        : one n(z) realization, all tomographic bins stacked into one vector
           of length N = Ntomo * Nz (bin-major: bin 0 first, then bin 1, ...)
nbar     : mean of the realizations
C_n      : covariance of the realizations                                (eq. 16)
J        : d(data vector)/d n, shape (Ndata, N)
F        : J^T C_dv^{-1} J, the Fisher matrix of the data vector w.r.t. n (eq. 10)
G        : Lambda_n^{1/2} V_n^T F V_n Lambda_n^{1/2}, F in the whitened
           n space, with C_n = V_n Lambda_n V_n^T                        (eq. 19)
lambda_G : eigenvalues of G, descending. Sum of the discarded ones is the
           mean chi^2 loss of the compression                            (eq. 21)
D        : decoder, n - nbar ~= D u,  shape (N, M)                       (eq. 23, 26)
E        : encoder, u = E (n - nbar), shape (M, N)                       (eq. 23, 25)
By construction the u of the input realizations have zero mean and unit
covariance, which is why a N(0,1) prior on each u_j is used in the chains.
"""
import numpy as np


# ----------------------------------------------------------------------------
# stacking helpers: (Ntomo, Nz) <-> length-N vector
# ----------------------------------------------------------------------------
def stack(nz):
    """(..., Ntomo, Nz) -> (..., Ntomo*Nz)"""
    nz = np.asarray(nz)
    return nz.reshape(nz.shape[:-2] + (-1,))


def unstack(n, ntomo):
    """(..., Ntomo*Nz) -> (..., Ntomo, Nz)"""
    n = np.asarray(n)
    return n.reshape(n.shape[:-1] + (ntomo, -1))


# ----------------------------------------------------------------------------
# step 1: statistics of the realizations
# ----------------------------------------------------------------------------
def mean_and_cov(samples):
    """samples: (Nsamples, N). Returns nbar (N,) and C_n (N, N) (unbiased)."""
    samples = np.asarray(samples, dtype="float64")
    nbar = samples.mean(axis=0)
    C_n = np.cov(samples, rowvar=False)
    return nbar, C_n


# ----------------------------------------------------------------------------
# step 2: Fisher matrix of the data vector with respect to n     (eq. 10)
# ----------------------------------------------------------------------------
def fisher_matrix(J, invcov):
    """F = J^T C_dv^{-1} J, symmetrized."""
    F = J.T @ invcov @ J
    return 0.5 * (F + F.T)


# ----------------------------------------------------------------------------
# step 3: the modes                                               (eqs. 19-26)
# ----------------------------------------------------------------------------
def compute_modes(C_n, F, rcond=1e-10):
    """
    Returns lambda_G (N,), D (N, N), E (N, N), for ALL modes, ordered by
    decreasing lambda_G. Truncate with D[:, :M], E[:M, :].

    rcond: eigenvalues of C_n below rcond * max are treated as zero, i.e.
    directions along which the realizations never vary (for example the
    normalization constraint). Their 1/sqrt(lambda_n) is set to zero (SVD
    pseudo-inverse convention, see the text after eq. 26).
    """
    C_n = 0.5 * (C_n + C_n.T)
    F = 0.5 * (F + F.T)

    # C_n = V_n Lambda_n V_n^T
    lambda_n, V_n = np.linalg.eigh(C_n)
    keep = lambda_n > rcond * lambda_n.max()
    sqrt_ln = np.zeros_like(lambda_n)
    inv_sqrt_ln = np.zeros_like(lambda_n)
    sqrt_ln[keep] = np.sqrt(lambda_n[keep])
    inv_sqrt_ln[keep] = 1.0 / sqrt_ln[keep]

    # G = Lambda_n^{1/2} V_n^T F V_n Lambda_n^{1/2}                      (eq. 19)
    W = V_n * sqrt_ln                    # = V_n Lambda_n^{1/2}, i.e. T^{-1}
    G = W.T @ F @ W
    G = 0.5 * (G + G.T)
    lambda_G, V_G = np.linalg.eigh(G)
    order = np.argsort(lambda_G)[::-1]   # descending
    lambda_G = np.maximum(lambda_G[order], 0.0)
    V_G = V_G[:, order]

    D = W @ V_G                          # V_n Lambda_n^{1/2} V_G           (eq. 23)
    E = V_G.T @ (V_n * inv_sqrt_ln).T    # V_G^T Lambda_n^{-1/2} V_n^T      (eq. 23)
    return lambda_G, D, E


def choose_nmodes(lambda_G, chi2_threshold):
    """Smallest M such that sum_{i>M} lambda_G < chi2_threshold.      (eq. 21)"""
    residual = np.cumsum(lambda_G[::-1])[::-1]          # residual[M] = sum_{i>=M}
    residual = np.append(residual, 0.0)                 # residual[N] = 0
    return int(np.argmax(residual < chi2_threshold))


def chi2_loss(lambda_G, nmodes):
    """Mean chi^2 lost by keeping only `nmodes` modes.               (eq. 21)"""
    return float(np.sum(lambda_G[nmodes:]))


# ----------------------------------------------------------------------------
# encoding / decoding of realizations
# ----------------------------------------------------------------------------
def encode(samples, nbar, E, nmodes):
    """u = E (n - nbar) for each sample. Returns (Nsamples, M)."""
    return (np.asarray(samples) - nbar) @ E[:nmodes, :].T


def decode(u, nbar, D):
    """n = nbar + D u. u: (..., M). Returns (..., N)."""
    u = np.asarray(u)
    M = u.shape[-1]
    return nbar + u @ D[:, :M].T


def chi2_between(dv_a, dv_b, invcov):
    """(dv_a - dv_b)^T C^{-1} (dv_a - dv_b)                              (eq. 3)"""
    d = np.asarray(dv_a) - np.asarray(dv_b)
    return float(d @ invcov @ d)
