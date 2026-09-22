"""
Step 3: validate the modes against the real Cosmolike data vector.

    python nz_pca/validate_modes.py --yaml EXAMPLE_NZ_MODES_FISHER.yaml \
           --realizations nz_realizations.npz --jacobian jacobian.npz \
           --modes data/DESY6_source_modes.txt [--nsamples 200] [--plot modes.png]

For a random subset of realizations n_a it computes, with Cosmolike,
  1. chi2 of the linearized model dv0 + J (n_a - nbar) against the true dv(n_a):
     how good the linearization behind the Fisher weighting is (should be << 1
     and much smaller than item 2 at M = 0);
  2. <chi2> between dv(n_a) and dv(n_hat_a), n_hat_a = nbar + D E (n_a - nbar),
     for M = 0, 1, 2, ... modes (Bernstein et al. eq. 3), compared with the
     linear prediction sum_{i>M} lambda_G (eq. 21).  This is the number the
     threshold (0.15 for DES Y6) refers to;
  3. the same for the shift-only model n_i(z - dz_i) with dz_i chosen to match
     the mean redshift of each n_a (the DES Y3-style alternative), for reference.
"""
import argparse
import sys
import os
import numpy as np

from cocoa_datavector import CocoaDataVector
from fisher_modes import stack, unstack, mean_and_cov, encode, decode
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "likelihood"))
from nz_modes import read_modes_file, interpolate_modes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--realizations", required=True)
    ap.add_argument("--jacobian", required=True)
    ap.add_argument("--modes", required=True, help="modes file written by compute_modes.py")
    ap.add_argument("--nsamples", type=int, default=200)
    ap.add_argument("--max-modes", type=int, default=10)
    ap.add_argument("--plot", default=None, help="write a png with the modes and the chi2 curves")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = CocoaDataVector(args.yaml)
    f = np.load(args.realizations)
    z_c, nz_real = f["z"], f["nz"]
    ntomo = nz_real.shape[1]
    samples = stack(nz_real)
    nbar, C_n = mean_and_cov(samples)
    jac = np.load(args.jacobian)
    J = jac["J"]
    summary = np.load(args.modes.replace(".txt", "") + "_summary.npz")
    lambda_G, D, E = summary["lambda_G"], summary["D"], summary["E"]
    nmax = min(args.max_modes, D.shape[1])

    def table_from_n(n):
        """fiducial .nz table + (n - nbar) interpolated to the fine grid (what the likelihood does)"""
        table = model.nz_table.copy()
        for i, d in enumerate(unstack(n - nbar, ntomo)):
            table[:, 1 + i] += np.interp(model.z, z_c, d, left=0.0, right=0.0)
        return table

    # the modes file must reproduce D (interpolation to the coarse grid is the identity)
    z_file, U_file = read_modes_file(args.modes)
    assert np.allclose(unstack(D[:, :U_file.shape[2]].T, ntomo).transpose(1, 2, 0), U_file)

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(samples.shape[0], size=min(args.nsamples, samples.shape[0]), replace=False)
    dv0 = model.dv_fiducial
    chi2_lin, chi2_mean = [], []
    chi2_modes = np.zeros((len(idx), nmax + 1))
    chi2_shift = []
    zbar = [np.trapz(z_c * unstack(nbar, ntomo)[i], z_c) for i in range(ntomo)]
    for a, s in enumerate(idx):
        n = samples[s]
        dv = model.datavector(table_from_n(n))
        chi2_lin.append(model.chi2(dv, dv0 + J @ (n - nbar)))
        chi2_mean.append(model.chi2(dv, dv0))
        u = encode(n[None, :], nbar, E, nmax)[0]
        for M in range(nmax + 1):
            dv_hat = model.datavector(table_from_n(decode(u[:M], nbar, D)))
            chi2_modes[a, M] = model.chi2(dv, dv_hat)
        dz = [np.trapz(z_c * unstack(n, ntomo)[i], z_c) - zbar[i] for i in range(ntomo)]
        chi2_shift.append(model.chi2(dv, model.datavector_with_shift(dz)))
    model.datavector(model.nz_table)

    chi2_lin, chi2_mean, chi2_shift = map(np.array, (chi2_lin, chi2_mean, chi2_shift))
    print(f"{len(idx)} realizations evaluated with Cosmolike")
    print(f"1. linearization error  <chi2(dv, dv0 + J dn)> = {chi2_lin.mean():.4f}  (max {chi2_lin.max():.4f})")
    print(f"   for reference       <chi2(dv, dv0)>        = {chi2_mean.mean():.3f}  (n(z) fixed to the mean)")
    print("2. compression loss <chi2(dv(n), dv(n_hat))>:  M   Cosmolike   linear prediction sum_{i>M} lambda_G")
    for M in range(nmax + 1):
        print(f"{M:47d}   {chi2_modes[:, M].mean():9.4f}   {lambda_G[M:].sum():9.4f}")
    print(f"3. shift-only model     <chi2(dv(n), dv(shifted nbar))> = {chi2_shift.mean():.4f}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        U = unstack(D[:, :nmax].T, ntomo)                       # (M, ntomo, Nz)
        for i in range(ntomo):
            for j in range(min(4, nmax)):
                axes[0].plot(z_c, U[j, i], color=f"C{j}", alpha=0.8, label=f"mode {j+1}" if i == 0 else None)
        axes[0].plot(z_c, unstack(nbar, ntomo).T / unstack(nbar, ntomo).max() * np.abs(U[:4]).max(),
                     color="gray", lw=0.7, ls="--")
        axes[0].set_xlabel("z"); axes[0].set_ylabel("U_ij(z)  (mean n(z) dashed, rescaled)")
        axes[0].set_title("first modes, all bins"); axes[0].legend()
        Ms = np.arange(nmax + 1)
        axes[1].semilogy(Ms, [lambda_G[M:].sum() for M in Ms], "o-", label="linear prediction")
        axes[1].semilogy(Ms, chi2_modes.mean(axis=0), "s--", label="Cosmolike")
        axes[1].axhline(0.15, color="gray", ls=":", label="DES Y6 threshold")
        axes[1].set_xlabel("number of modes M"); axes[1].set_ylabel("<chi2> loss"); axes[1].legend()
        axes[1].set_title("compression loss")
        bins = np.linspace(0, np.percentile(chi2_mean, 98), 30)
        axes[2].hist(chi2_mean, bins, histtype="step", label="original realizations")
        Mshow = min(nmax, int(summary["nmodes"]))
        chi2_rec = [model.chi2(model.datavector(table_from_n(decode(encode(samples[s][None], nbar, E, Mshow)[0], nbar, D))), dv0)
                    for s in idx]
        model.datavector(model.nz_table)
        axes[2].hist(chi2_rec, bins, histtype="step", label=f"reconstructed, M={Mshow}")
        axes[2].set_xlabel("chi2(dv(n), dv(nbar))"); axes[2].legend(); axes[2].set_title("Yin et al. Fig. E3 analogue")
        fig.tight_layout(); fig.savefig(args.plot, dpi=130)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
