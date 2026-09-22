"""
Unit tests of the pure-numpy parts (no Cosmolike needed):  pytest nz_pca/test_fisher_modes.py

A synthetic problem with a LINEAR data vector dv = J n (so the Fisher-weighted
PCA is exact) checks the algebra of Bernstein et al. 2025 directly:
  - u = E (n - nbar) has zero mean and unit covariance, E D = I
  - the compression loss of eq. (21) equals the directly computed <chi2>
  - truncation picks the smallest M under the threshold
  - modes file round trip, interpolation, and NzModes.nz()
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "likelihood"))
from fisher_modes import (stack, unstack, mean_and_cov, fisher_matrix, compute_modes, choose_nmodes,
                          chi2_loss, encode, decode, chi2_between)
from nz_modes import NzModes, read_modes_file, write_modes_file, interpolate_modes


def synthetic_problem(seed=0, nsamples=3000, ntomo=3, nz=25, ndata=40):
    rng = np.random.default_rng(seed)
    z = np.linspace(0, 2.4, nz)
    # realizations: a few smooth random modes on top of a mean, correlated across bins
    nbar_true = np.stack([np.exp(-0.5 * ((z - 0.5 - 0.4 * i) / 0.25) ** 2) for i in range(ntomo)])
    basis = np.stack([np.sin((k + 1) * np.pi * z / 2.4) for k in range(6)])       # (6, nz)
    amps = rng.normal(size=(nsamples, 6)) * np.array([0.05, 0.03, 0.02, 0.01, 0.005, 0.002])
    nz_real = nbar_true[None] + np.einsum("sk,kz->sz", amps, basis)[:, None, :] * np.array([1.0, 0.7, -0.5])[None, :, None]
    nz_real += rng.normal(scale=1e-4, size=nz_real.shape)                       # tiny white noise
    samples = stack(nz_real)
    J = rng.normal(size=(ndata, ntomo * nz)) * np.linspace(1, 0.1, ntomo * nz)  # linear model dv = J n
    cov = np.diag(rng.uniform(0.5, 2.0, ndata))
    return z, nz_real, samples, J, np.linalg.inv(cov)


def test_encoder_decoder_algebra():
    z, nz_real, samples, J, invcov = synthetic_problem()
    nbar, C_n = mean_and_cov(samples)
    lambda_G, D, E = compute_modes(C_n, fisher_matrix(J, invcov))
    M = 6
    u = encode(samples, nbar, E, M)
    assert np.abs(u.mean(axis=0)).max() < 1e-10
    assert np.abs(np.cov(u, rowvar=False) - np.eye(M)).max() < 1e-8
    assert np.allclose(E[:M] @ D[:, :M], np.eye(M), atol=1e-8)
    assert np.all(np.diff(lambda_G) <= 1e-12)                       # descending


def test_chi2_loss_matches_direct_computation():
    z, nz_real, samples, J, invcov = synthetic_problem()
    nbar, C_n = mean_and_cov(samples)
    F = fisher_matrix(J, invcov)
    lambda_G, D, E = compute_modes(C_n, F)
    for M in [0, 1, 3, 6]:
        u = encode(samples, nbar, E, M)
        n_hat = decode(u, nbar, D)
        # dv is linear: chi2 = (n - n_hat)^T F (n - n_hat)
        chi2 = np.array([chi2_between(J @ n, J @ nh, invcov) for n, nh in zip(samples, n_hat)])
        # sample covariance is unbiased with N-1 -> tiny mismatch with the population mean
        assert abs(chi2.mean() - chi2_loss(lambda_G, M)) < 2e-3 * max(1.0, chi2_loss(lambda_G, M)) + 1e-8


def test_choose_nmodes():
    lam = np.array([10.0, 1.0, 0.1, 0.01, 0.0])
    assert choose_nmodes(lam, 0.15) == 2          # loss after 2 modes = 0.11 < 0.15
    assert choose_nmodes(lam, 0.1) == 3           # loss after 3 modes = 0.01
    assert choose_nmodes(lam, 2.0) == 1           # loss after 1 mode = 1.11
    assert choose_nmodes(lam, 100.0) == 0
    assert choose_nmodes(lam, 1e-9) == 4          # loss after 4 modes = 0.0 < 1e-9


def test_modes_file_roundtrip_and_nz(tmp_path):
    z, nz_real, samples, J, invcov = synthetic_problem()
    ntomo, nz = nz_real.shape[1:]
    nbar, C_n = mean_and_cov(samples)
    lambda_G, D, E = compute_modes(C_n, fisher_matrix(J, invcov))
    U = unstack(D[:, :4].T, ntomo).transpose(1, 2, 0)              # (ntomo, nz, 4)
    fname = str(tmp_path / "modes.txt")
    write_modes_file(fname, z, U)
    z2, U2 = read_modes_file(fname)
    assert np.allclose(z, z2) and np.allclose(U, U2, rtol=1e-9, atol=1e-14)

    # NzModes on a finer grid: n = nbar_fine + sum u_j interp(U_j)
    z_fine = np.linspace(0, 2.4, 97)
    table = np.column_stack([z_fine] + [np.interp(z_fine, z, unstack(nbar, ntomo)[i]) for i in range(ntomo)])
    modes = NzModes(table, fname, nmodes=3)
    assert np.array_equal(modes.nz([0, 0, 0]), table)
    u = np.array([1.5, -0.5, 2.0])
    out = modes.nz(u)
    expected = table.copy()
    expected[:, 1:] += (interpolate_modes(z, U[:, :, :3], z_fine) @ u).T
    assert np.allclose(out, expected)
    assert out is not modes.nbar and not np.shares_memory(out, modes.nbar)   # fresh array every call


def test_stack_unstack():
    a = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    assert np.array_equal(unstack(stack(a), 3), a)
    assert stack(a).shape == (2, 12)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
