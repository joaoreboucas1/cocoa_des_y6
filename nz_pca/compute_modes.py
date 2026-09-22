"""
Step 2 of the n(z) mode construction: from the realizations and the Jacobian
(compute_jacobian.py) build the Fisher-weighted modes and write the modes
file read by the likelihood.

    python nz_pca/compute_modes.py --realizations nz_realizations.npz \
           --jacobian jacobian.npz --out data/DESY6_source_modes.txt

Follows Bernstein et al. 2025 (arXiv:2506.00758) steps 1-5:
  1. nbar and C_n of the realizations; F = J^T C_dv^{-1} J
  2. G = Lambda_n^{1/2} V_n^T F V_n Lambda_n^{1/2}, eigenvalues lambda_G descending
  3. M = smallest number of modes with sum_{i>M} lambda_G < chi2_threshold
     (DES Y6 sources: 0.15 -> 7 modes)
  4. decoder D (the modes) and encoder E
  5. u = E (n - nbar) of every realization: must have mean 0 and covariance 1
The modes file contains the first --max-modes modes; the likelihood uses the
first `source_nz_nmodes` of them.
"""
import argparse
import sys
import os
import numpy as np

from fisher_modes import stack, unstack, mean_and_cov, fisher_matrix, compute_modes, \
    choose_nmodes, chi2_loss, encode
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "likelihood"))
from nz_modes import write_modes_file


def load_jacobian(files):
    """Merge the output(s) of compute_jacobian.py (possibly chunked)."""
    J, computed, ref = None, None, None
    for f in files:
        d = np.load(f)
        if J is None:
            J, computed, ref = np.zeros_like(d["J"]), np.zeros_like(d["computed"]), d
        J += d["J"]
        computed |= d["computed"]
    if not computed.all():
        raise ValueError(f"Jacobian columns missing: {np.flatnonzero(~computed)}")
    return J, ref["invcov"], ref["z"], ref["nbar"], int(ref["ntomo"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--realizations", required=True, help=".npz with z and nz (Nsamples, Ntomo, Nz)")
    ap.add_argument("--jacobian", required=True, nargs="+", help="output(s) of compute_jacobian.py")
    ap.add_argument("--out", required=True, help="modes file for the likelihood (text)")
    ap.add_argument("--chi2-threshold", type=float, default=0.15, help="mean chi2 loss allowed (default 0.15)")
    ap.add_argument("--nmodes", type=int, default=None, help="force this number of modes instead")
    ap.add_argument("--max-modes", type=int, default=20, help="how many modes to write to the file")
    args = ap.parse_args()

    f = np.load(args.realizations)
    z, nz_real = f["z"], f["nz"]
    samples = stack(nz_real)                       # (Nsamples, Ntomo*Nz)
    nbar, C_n = mean_and_cov(samples)
    ntomo = nz_real.shape[1]

    J, invcov, z_J, nbar_J, ntomo_J = load_jacobian(args.jacobian)
    if not (np.allclose(z, z_J) and ntomo == ntomo_J and np.allclose(nbar, nbar_J)):
        raise ValueError("the Jacobian was computed for different realizations")

    F = fisher_matrix(J, invcov)
    lambda_G, D, E = compute_modes(C_n, F)

    M = args.nmodes if args.nmodes is not None else choose_nmodes(lambda_G, args.chi2_threshold)
    nwrite = min(args.max_modes, np.count_nonzero(lambda_G > 0))

    print(f"{samples.shape[0]} realizations, {ntomo} bins x {len(z)} z points")
    print(f"<chi2> if n(z) were fixed to the mean (no modes): {chi2_loss(lambda_G, 0):.3f}")
    print(" mode   lambda_G   <chi2> lost if truncated after this mode")
    for i in range(nwrite):
        print(f"{i+1:5d} {lambda_G[i]:10.4f} {chi2_loss(lambda_G, i+1):10.4f}" + ("   <-- M" if i + 1 == M else ""))
    print(f"chosen number of modes M = {M}  (threshold {args.chi2_threshold}, loss {chi2_loss(lambda_G, M):.4f})")

    # sanity: u of the input realizations has mean 0 and covariance 1
    u = encode(samples, nbar, E, M)
    cov_u = np.cov(u, rowvar=False) if M > 1 else np.atleast_2d(np.var(u))
    print(f"u of the realizations: |mean| < {np.abs(u.mean(axis=0)).max():.1e}, "
          f"|cov - 1| < {np.abs(cov_u - np.eye(M)).max():.1e}")
    for j in range(M):
        x = u[:, j]
        print(f"   u_{j+1}: std {x.std():.3f}, skewness {np.mean(x**3):+.3f}, "
              f"kurtosis-3 {np.mean(x**4)-3:+.3f}, fraction |u|>3: {np.mean(np.abs(x) > 3):.4f}")

    U = unstack(D[:, :nwrite].T, ntomo).transpose(1, 2, 0)      # (Ntomo, Nz, nwrite)
    write_modes_file(args.out, z, U,
                     header=f"n(z) modes, Fisher-weighted PCA (Bernstein et al. 2025). recommended M = {M} "
                            f"(chi2 threshold {args.chi2_threshold}); lambda_G = " +
                            " ".join(f"{x:.4g}" for x in lambda_G[:nwrite]))
    np.savez(args.out.replace(".txt", "") + "_summary.npz", z=z, nbar=nbar, lambda_G=lambda_G,
             D=D[:, :nwrite], E=E[:nwrite], u=u, nmodes=M, ntomo=ntomo)
    print(f"wrote {args.out} ({nwrite} modes) and the _summary.npz file")


if __name__ == "__main__":
    main()
