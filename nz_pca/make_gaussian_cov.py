"""
Gaussian (shape noise + cosmic variance) covariance of the cosmic-shear data
vector, in cocoa's `i j C_ij` format.  TESTING AID ONLY: the analysis uses the
DES Y6 covariance from CosmoCov; this one lets the n(z)-mode pipeline be
exercised with a covariance of realistic size before that file is available.

    python nz_pca/make_gaussian_cov.py --yaml EXAMPLE_NZ_MODES_FISHER.yaml \
           --out data/gaussian_xi.cov --out-datavector data/fiducial_xi.modelvector

The C_ell come from Cosmolike at the fiducial point of the yaml.  Because the
likelihood needs an existing covariance file just to initialize, the yaml's
.dataset is replaced by --bootstrap-dataset (default DESY6.dataset, whose
dummy.cov / dummy.modelvector ship with the repo) while this script runs; the
fiducial theory vector is written too, so both files that
DESY6_nzpca_test.dataset needs come out of this one step.  Formula
(e.g. Joachimi, Schneider & Eifler 2008), with theta-bin-averaged Bessel
functions:
  Cov[xi_a^{ij}(t1), xi_b^{kl}(t2)] = 1/(2 pi A) int dl l Jbar_a(l,t1) Jbar_b(l,t2)
        [ (C_ik + N_ik)(C_jl + N_jl) + (C_il + N_il)(C_jk + N_jk) ]
with N_ij = delta_ij sigma_e^2 / n_i.  The non-shear part of the 1000-long
3x2pt vector gets a unit diagonal (it is masked out anyway).
"""
import argparse
import numpy as np
from scipy.special import j1, jv

from cocoa_datavector import CocoaDataVector
import cosmolike_des_y6_interface as ci

ARCMIN = np.pi / 180.0 / 60.0


def bessel_bin_averaged(ell, tmin, tmax):
    """Averages of J0 and J4 over the annulus [tmin, tmax] (weights ~ theta)."""
    def F0(x):   # int x J0(x) dx = x J1(x)
        return x * j1(x)

    def F4(x):   # int x J4(x) dx = (x - 8/x) J1(x) - 8 J2(x)
        return (x - 8.0 / x) * j1(x) - 8.0 * jv(2, x)
    norm = 2.0 / (ell**2 * (tmax**2 - tmin**2))
    return norm * (F0(ell * tmax) - F0(ell * tmin)), norm * (F4(ell * tmax) - F4(ell * tmin))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--out", required=True, help="covariance, cocoa `i j C_ij` format")
    ap.add_argument("--out-datavector", default=None,
                    help="also write the fiducial theory vector in cocoa `i value` format")
    ap.add_argument("--bootstrap-dataset", default="DESY6.dataset",
                    help=".dataset (with existing files) used to initialize the likelihood")
    ap.add_argument("--area-deg2", type=float, default=4143.0, help="DES Y6 footprint")
    ap.add_argument("--sigma-e", type=float, default=0.27, help="shape noise per component")
    ap.add_argument("--neff", type=float, nargs="+", default=[1.5, 1.5, 1.5, 1.5], help="gal/arcmin^2 per bin")
    ap.add_argument("--ellmax", type=float, default=5e4)
    args = ap.parse_args()

    model = CocoaDataVector(args.yaml, likelihood_options={"data_file": args.bootstrap_dataset})
    like = model.like
    ntomo, ntheta = model.ntomo, like.ntheta
    pairs = [(i, j) for i in range(ntomo) for j in range(i, ntomo)]   # cocoa ordering of xi bin pairs

    # theta bin edges (log-spaced, as in Cosmolike's init_binning)
    edges = np.geomspace(like.theta_min_arcmin, like.theta_max_arcmin, ntheta + 1) * ARCMIN

    # C_ell on a dense grid (Cosmolike is queried on a coarse log grid, then interpolated)
    ell_coarse = np.geomspace(1.0, args.ellmax, 400)
    EE, _ = ci.C_ss_tomo_limber(ell_coarse)                  # (Nell, ntomo, ntomo), upper triangle filled
    ell = np.arange(1.0, args.ellmax + 1.0)
    C = np.zeros((ntomo, ntomo, ell.size))
    for i, j in pairs:
        C[i, j] = C[j, i] = np.exp(np.interp(np.log(ell), np.log(ell_coarse), np.log(np.abs(EE[:, i, j]) + 1e-300)))
    noise = args.sigma_e**2 / (np.array(args.neff) / ARCMIN**2)   # sr
    for i in range(ntomo):
        C[i, i] += noise[i]

    J = np.zeros((2, ntheta, ell.size))                       # bin-averaged J0 (xi+) and J4 (xi-)
    for t in range(ntheta):
        J[0, t], J[1, t] = bessel_bin_averaged(ell, edges[t], edges[t + 1])

    area = args.area_deg2 * (np.pi / 180.0) ** 2
    npair = len(pairs)
    cov = np.zeros((2, npair, ntheta, 2, npair, ntheta))
    for p, (i, j) in enumerate(pairs):
        for q, (k, l) in enumerate(pairs):
            kernel = ell * (C[i, k] * C[j, l] + C[i, l] * C[j, k]) / (2.0 * np.pi * area)
            for a in range(2):
                for b in range(2):
                    cov[a, p, :, b, q, :] = (J[a] * kernel) @ J[b].T
    nxi = 2 * npair * ntheta
    cov = cov.reshape(nxi, nxi)

    ndata = model.dv_fiducial.size
    full = np.eye(ndata)
    full[:nxi, :nxi] = cov
    ii, jj = np.meshgrid(np.arange(ndata), np.arange(ndata), indexing="ij")
    np.savetxt(args.out, np.column_stack([ii.ravel(), jj.ravel(), full.ravel()]), fmt="%d %d %.10e")
    sn = np.sqrt(model.dv_fiducial[:nxi] @ np.linalg.solve(cov, model.dv_fiducial[:nxi]))
    print(f"wrote {args.out}; xi+- block {nxi}x{nxi}; S/N of the fiducial xi+- = {sn:.1f}")
    if args.out_datavector:
        np.savetxt(args.out_datavector, np.column_stack([np.arange(ndata), model.dv_fiducial]), fmt="%d %.8e")
        print(f"wrote {args.out_datavector} (fiducial theory vector, {ndata} entries)")


if __name__ == "__main__":
    main()
