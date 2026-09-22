"""
Step 1 of the n(z) mode construction: the Jacobian J = d(data vector)/d n(z_k)
of the cocoa/Cosmolike data vector, one column per (tomographic bin, z point)
of the n(z) realizations, at the fiducial point of a cobaya yaml file.

    python nz_pca/compute_jacobian.py --yaml EXAMPLE_NZ_FISHER.yaml \
           --realizations nz_realizations.npz --out jacobian.npz

Derivatives are computed with derivkit (arXiv:2602.08078): the data vector is
evaluated at n_points+1 values of n(z_k) spread over +- step_sigma * sigma_k
around the mean (sigma_k = scatter of the realizations at that grid point),
and a low-order polynomial fit gives the slope.  Nothing has to be tuned by
hand: the window is set by how much the realizations actually vary there,
and the fit averages over the numerical roughness of the Limber integrals.

The perturbation of one grid point of the (coarse) realization grid is
linearly interpolated onto the (fine) grid of the .nz file before it is given
to Cosmolike, which is what the likelihood does with the modes later on, so
J describes exactly the model that will be sampled.

Cost: (n_points + 1) Cosmolike evaluations per grid point.  For long jobs use
--chunk i/n in n independent jobs and give all the outputs to compute_modes.py.

Realizations file: .npz with  z  (Nz,)  and  nz  (Nsamples, Ntomo, Nz).
"""
import argparse
import time
import numpy as np
from derivkit import DerivativeKit

from cocoa_datavector import CocoaDataVector
from fisher_modes import stack, unstack, mean_and_cov


def load_realizations(filename):
    f = np.load(filename)
    z, nz = f["z"], f["nz"]
    if nz.ndim != 3:
        raise ValueError(f"{filename}: nz must have shape (Nsamples, Ntomo, Nz)")
    return z, nz


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yaml", required=True, help="cobaya yaml with an evaluate override = fiducial point")
    ap.add_argument("--realizations", required=True, help=".npz with z (Nz,) and nz (Nsamples, Ntomo, Nz)")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--method", default="adaptive", choices=["adaptive", "finite"],
                    help="derivkit method (default adaptive polynomial fit; finite = 5-point stencil)")
    ap.add_argument("--n-points", type=int, default=6, help="adaptive: sample points (evaluations = n_points+1)")
    ap.add_argument("--step-sigma", type=float, default=2.0,
                    help="half-width of the sampling window in units of sigma_k (default 2)")
    ap.add_argument("--chunk", default=None, help="i/n: compute only columns k with k %% n == i")
    args = ap.parse_args()

    model = CocoaDataVector(args.yaml)
    z_c, nz_real = load_realizations(args.realizations)
    ntomo = nz_real.shape[1]
    if ntomo != model.ntomo:
        raise ValueError(f"realizations have {ntomo} bins, the .nz file has {model.ntomo}")
    nbar, C_n = mean_and_cov(stack(nz_real))          # on the coarse grid, stacked
    sigma = np.sqrt(np.diag(C_n))
    N = nbar.size
    ndata = model.dv_fiducial.size

    def table_from_delta(delta):
        """.nz table = fiducial table + delta (coarse, stacked) interpolated to the fine grid"""
        table = model.nz_table.copy()
        for i, d in enumerate(unstack(delta, ntomo)):
            table[:, 1 + i] += np.interp(model.z, z_c, d, left=0.0, right=0.0)
        return table

    columns = np.arange(N)
    if args.chunk:
        i, n = (int(s) for s in args.chunk.split("/"))
        columns = columns[columns % n == i]

    # grid points where the realizations never vary carry no information: skip
    active = sigma > 1e-6 * sigma.max()
    step_floor = 1e-3 * np.abs(nbar).max()

    J = np.zeros((ndata, N))
    fit_error = np.zeros((ndata, N))
    computed = np.zeros(N, dtype=bool)
    t0 = time.time()
    for count, k in enumerate(columns):
        computed[k] = True
        if not active[k]:
            continue

        def dv_of_nk(x, k=k):
            delta = np.zeros(N)
            delta[k] = x - nbar[k]
            return model.datavector(table_from_delta(delta))

        h = max(args.step_sigma * sigma[k], step_floor)
        dk = DerivativeKit(dv_of_nk, x0=float(nbar[k]))
        if args.method == "adaptive":
            J[:, k], fit_error[:, k] = dk.differentiate(order=1, method="adaptive",
                                                        n_points=args.n_points, spacing=h,
                                                        return_error=True)
        else:
            J[:, k] = dk.differentiate(order=1, method="finite", stepsize=h / 2, num_points=5)

        if count % 10 == 0:
            b, iz = divmod(k, len(z_c))
            print(f"column {k:4d}/{N} (bin {b}, z={z_c[iz]:.3f}, h={h:.2e})  "
                  f"{time.time()-t0:.0f}s elapsed", flush=True)

    model.datavector(model.nz_table)   # leave Cosmolike at the fiducial n(z)

    out = args.out
    if args.chunk:
        out = out.replace(".npz", "") + f"_chunk{i}of{n}.npz"
    np.savez(out, J=J, fit_error=fit_error, computed=computed, z=z_c, nbar=nbar, ntomo=ntomo,
             invcov=model.invcov, mask=model.mask, dv_fiducial=model.dv_fiducial,
             method=args.method, n_points=args.n_points, step_sigma=args.step_sigma,
             yaml=args.yaml, realizations=args.realizations)
    print(f"wrote {out}: J shape {J.shape}, {computed.sum()} columns, "
          f"{(computed & ~active).sum()} skipped (no scatter), {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
