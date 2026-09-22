"""
Dummy n(z) realizations, to exercise the pipeline until the DES Y6
SOMPZ+WZ+BL realizations are available.

    python nz_pca/make_dummy_realizations.py --nz data/DESY6_source.nz \
           --out data/dummy_nz_realizations.npz

Each realization is the fiducial n(z) of every bin shifted, stretched, and
modulated by a smooth random field, plus a small random bump (outlier
population); bins are correlated through a shared shift.  The realizations
are clipped at zero and normalized to unit integral.  Grid: DES-like nodes
z = 0, 0.05, ..., 3.  Output .npz: z (Nz,), nz (Nsamples, Ntomo, Nz).
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nz", required=True, help="cocoa .nz file with the fiducial n(z) (z, n_1(z), ...)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nsamples", type=int, default=2000)
    ap.add_argument("--dz-grid", type=float, default=0.05)
    ap.add_argument("--zmax", type=float, default=3.0)
    ap.add_argument("--sigma-shift", type=float, default=0.01, help="per-bin shift scatter")
    ap.add_argument("--sigma-shift-common", type=float, default=0.005, help="shift common to all bins")
    ap.add_argument("--sigma-stretch", type=float, default=0.03)
    ap.add_argument("--sigma-field", type=float, default=0.05, help="amplitude of the smooth modulation")
    ap.add_argument("--bump-fraction", type=float, default=0.01, help="mean outlier fraction")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    table = np.loadtxt(args.nz)
    z_fine, nbar_fine = table[:, 0], table[:, 1:].T          # (Ntomo, Nz_fine)
    ntomo = nbar_fine.shape[0]
    z = np.arange(0.0, args.zmax + 1e-9, args.dz_grid)
    nz = len(z)

    # smooth random field: Gaussian correlation, length 0.15 in z
    corr = np.exp(-0.5 * ((z[:, None] - z[None, :]) / 0.15) ** 2) + 1e-8 * np.eye(nz)
    L = np.linalg.cholesky(corr)

    out = np.zeros((args.nsamples, ntomo, nz))
    for r in range(args.nsamples):
        common = rng.normal(0.0, args.sigma_shift_common)
        for i in range(ntomo):
            shift = common + rng.normal(0.0, args.sigma_shift)
            stretch = rng.normal(1.0, args.sigma_stretch)
            zmean = np.trapz(z_fine * nbar_fine[i], z_fine) / np.trapz(nbar_fine[i], z_fine)
            # n((z - zmean)/stretch + zmean - shift): stretch about the mean, then shift
            zz = (z - zmean) / stretch + zmean - shift
            n = np.interp(zz, z_fine, nbar_fine[i], left=0.0, right=0.0)
            n = n * (1.0 + args.sigma_field * (L @ rng.normal(size=nz)))
            zb = rng.uniform(0.1, args.zmax - 0.1)
            bump = np.exp(-0.5 * ((z - zb) / 0.1) ** 2)
            n = n + rng.exponential(args.bump_fraction) * bump / np.trapz(bump, z)
            n = np.clip(n, 0.0, None)
            out[r, i] = n / np.trapz(n, z)

    np.savez(args.out, z=z, nz=out)
    mean_z = [np.trapz(z * out[:, i].mean(axis=0), z) for i in range(ntomo)]
    print(f"wrote {args.out}: {args.nsamples} realizations, {ntomo} bins, {nz} z points; "
          f"<z> per bin = {np.round(mean_z, 3)}")


if __name__ == "__main__":
    main()
