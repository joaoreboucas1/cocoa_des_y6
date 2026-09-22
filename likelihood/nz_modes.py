"""
Mode-projection ("n(z) PCA") model of the redshift distributions, as used by
DES Y6 (Bernstein et al. 2025, arXiv:2506.00758; Yin et al. 2025,
arXiv:2510.23566, eq. 30):

    n_i(z) = nbar_i(z) + sum_{j=1}^{M} u_j U_ij(z)

  nbar_i(z) : mean of the n(z) realizations of tomographic bin i (this is the
              table cocoa reads from `nz_source_file` in the .dataset file)
  U_ij(z)   : mode j evaluated in bin i (one amplitude u_j moves ALL bins at
              once, so bin-to-bin correlations are kept)
  u_j       : sampled amplitudes, prior N(0,1) truncated at +-3 in DES Y6

This file contains only what the likelihood needs at run time: reading the
modes file and evaluating the formula above.  The modes themselves are built
by the scripts in `projects/des_y6/nz_pca/` (see the README there).

MODES FILE FORMAT (plain text, written by nz_pca/compute_modes.py):
    one row per (bin, z) pair, bin-major (all z of bin 0, then all z of bin 1, ...)
    columns:  bin   z   U_1(z)   U_2(z)  ...  U_Mmax(z)
The z grid of the modes file does not need to be the z grid of the .nz file:
modes are linearly interpolated (zero outside their range), which is the same
"triangular kernel" rebinning DES uses to go from dz=0.05 to dz=0.01.
"""
import numpy as np


def read_modes_file(filename):
    """Returns z (Nz,), U (Ntomo, Nz, Mmax)."""
    table = np.loadtxt(filename)
    bins = table[:, 0].astype(int)
    ntomo = bins.max() + 1
    nz = np.count_nonzero(bins == 0)
    if table.shape[0] != ntomo * nz:
        raise ValueError(f"{filename}: expected {ntomo} bins x {nz} z values, "
                         f"found {table.shape[0]} rows")
    z = table[:nz, 1]
    U = table[:, 2:].reshape(ntomo, nz, -1)
    return z, U


def write_modes_file(filename, z, U, header=""):
    """Inverse of read_modes_file. U has shape (Ntomo, Nz, M)."""
    ntomo, nz, nmodes = U.shape
    rows = np.zeros((ntomo * nz, 2 + nmodes))
    rows[:, 0] = np.repeat(np.arange(ntomo), nz)
    rows[:, 1] = np.tile(z, ntomo)
    rows[:, 2:] = U.reshape(ntomo * nz, nmodes)
    header = (header + "\n" if header else "") + \
        "columns: bin  z  U_1(z) ... U_M(z)   (rows are bin-major)"
    np.savetxt(filename, rows, header=header, fmt=["%d", "%.6f"] + ["%.10e"] * nmodes)


def interpolate_modes(z_modes, U, z_new):
    """Linear interpolation of U (Ntomo, Nz, M) from z_modes to z_new; zero outside."""
    ntomo, _, nmodes = U.shape
    out = np.zeros((ntomo, len(z_new), nmodes))
    for i in range(ntomo):
        for j in range(nmodes):
            out[i, :, j] = np.interp(z_new, z_modes, U[i, :, j], left=0.0, right=0.0)
    return out


class NzModes:
    """
    Evaluates n_i(z) = nbar_i(z) + sum_j u_j U_ij(z) in the cocoa table format,
    i.e. an array of shape (Nz, 1 + Ntomo) whose first column is z.
    """

    def __init__(self, nz_table, modes_file, nmodes):
        self.nbar = np.array(nz_table, dtype="float64")       # (Nz, 1+Ntomo)
        self.z = self.nbar[:, 0]
        self.ntomo = self.nbar.shape[1] - 1
        z_modes, U = read_modes_file(modes_file)
        if U.shape[0] != self.ntomo:
            raise ValueError(f"{modes_file} has {U.shape[0]} bins, .nz file has {self.ntomo}")
        if nmodes > U.shape[2]:
            raise ValueError(f"{nmodes} modes requested, {modes_file} has only {U.shape[2]}")
        self.nmodes = nmodes
        # (Ntomo, Nz, M) on the .nz grid; the interpolation is done once, here
        self.U = interpolate_modes(z_modes, U[:, :, :nmodes], self.z)

    def nz(self, u):
        """u: sequence of M amplitudes. Returns a new (Nz, 1+Ntomo) table.
        (A new array every call on purpose: ci.set_source_sample takes ownership
        of the numpy buffer it receives, so it must never be given self.nbar.)"""
        u = np.asarray(u, dtype="float64")
        if u.shape != (self.nmodes,):
            raise ValueError(f"expected {self.nmodes} amplitudes, got {u.shape}")
        table = self.nbar.copy()
        table[:, 1:] += (self.U @ u).T                      # (Ntomo, Nz) -> (Nz, Ntomo)
        return table
