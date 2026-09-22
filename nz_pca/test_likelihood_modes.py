"""
End-to-end test of the likelihood side (source_nz_nmodes > 0 in cobaya).

    python nz_pca/test_likelihood_modes.py --yaml EXAMPLE_NZ_MODES_FISHER.yaml \
           --realizations nz_realizations.npz --modes data/DESY6_source_modes.txt

Runs `cobaya-run` (evaluate sampler, print_datavector) in separate processes
(Cosmolike keeps global state, so one likelihood per process) and checks:
  A. modes on, all u_j = 0        == modes off (external_nz_modeling 0 and 1)
  B. modes on, u = E (n_a - nbar) == the data vector of the reconstructed
                                     realization computed directly with
                                     CocoaDataVector (same numbers, two code paths)
  C. a "shift mode" U_i(z) = -d nbar_i/dz with u_1 = dz reproduces Cosmolike's
     own photo-z shift DES_DZ_S* = dz to first order in dz (sign convention)
"""
import argparse
import os
import subprocess
import sys
import tempfile
import numpy as np
import yaml

from fisher_modes import stack, unstack, mean_and_cov, encode, decode
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "likelihood"))
from nz_modes import read_modes_file, write_modes_file, interpolate_modes


def run_evaluate(base_info, like_opts, param_values, out_prefix):
    """cobaya-run with modified likelihood options / override values; returns the data vector."""
    info = yaml.safe_load(yaml.safe_dump(base_info))         # deep copy
    like_name = list(info["likelihood"])[0]
    info["likelihood"][like_name].update(like_opts)
    info["likelihood"][like_name]["print_datavector"] = True
    info["likelihood"][like_name]["print_datavector_file"] = out_prefix + ".modelvector"
    for name, value in param_values.items():           # fix these parameters at the given values
        info["params"][name] = {"value": float(value)}
        info["sampler"]["evaluate"]["override"].pop(name, None)
    info["output"] = out_prefix
    with open(out_prefix + ".yaml", "w") as f:
        yaml.safe_dump(info, f, sort_keys=False)
    r = subprocess.run(["cobaya-run", "-f", out_prefix + ".yaml"], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise RuntimeError(f"cobaya-run failed for {out_prefix}")
    return np.loadtxt(out_prefix + ".modelvector")[:, 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--realizations", required=True)
    ap.add_argument("--modes", required=True)
    ap.add_argument("--nmodes", type=int, default=6)
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()

    workdir = args.workdir or tempfile.mkdtemp(prefix="nz_modes_test_")
    base = yaml.safe_load(open(args.yaml))
    like_name = list(base["likelihood"])[0]
    data_dir = os.path.join(base["likelihood"][like_name]["path"])
    dataset = os.path.join(data_dir, base["likelihood"][like_name]["data_file"])
    modes_file_in_dataset = [l.split("=")[1].strip() for l in open(dataset) if l.startswith("nz_source_modes_file")][0]
    assert os.path.samefile(os.path.join(data_dir, modes_file_in_dataset), args.modes), \
        "the .dataset file of the yaml must point to --modes"
    M = args.nmodes
    u_names = [f"DES_U_S{j+1}" for j in range(M)]

    # ---- A: u = 0 reproduces the no-modes data vector -------------------------
    dv_off0 = run_evaluate(base, {"external_nz_modeling": 0, "source_nz_nmodes": 0}, {}, f"{workdir}/A_off0")
    dv_off1 = run_evaluate(base, {"external_nz_modeling": 1, "source_nz_nmodes": 0}, {}, f"{workdir}/A_off1")
    dv_u0 = run_evaluate(base, {"external_nz_modeling": 1, "source_nz_nmodes": M},
                         {n: 0.0 for n in u_names}, f"{workdir}/A_u0")
    print(f"A. max|dv(modes, u=0) - dv(no modes)| = {np.abs(dv_u0 - dv_off1).max():.3e}   "
          f"(external_nz_modeling 0 vs 1: {np.abs(dv_off0 - dv_off1).max():.3e})")
    assert np.array_equal(dv_u0, dv_off1) and np.array_equal(dv_off0, dv_off1)

    # ---- B: a reconstructed realization, two code paths ------------------------
    f = np.load(args.realizations)
    z_c, nz_real = f["z"], f["nz"]
    ntomo = nz_real.shape[1]
    samples = stack(nz_real)
    nbar, _ = mean_and_cov(samples)
    summary = np.load(args.modes.replace(".txt", "") + "_summary.npz")
    D, E = summary["D"], summary["E"]
    a = int(np.argmax(np.sum((samples - nbar) ** 2, axis=1)))      # the most extreme realization
    u = encode(samples[a][None], nbar, E, M)[0]
    dv_like = run_evaluate(base, {"external_nz_modeling": 1, "source_nz_nmodes": M},
                           {n: float(v) for n, v in zip(u_names, u)}, f"{workdir}/B_u")
    from cocoa_datavector import CocoaDataVector
    model = CocoaDataVector(f"{workdir}/A_off1.yaml")
    table = model.nz_table.copy()
    for i, d in enumerate(unstack(decode(u, nbar, D) - nbar, ntomo)):
        table[:, 1 + i] += np.interp(model.z, z_c, d, left=0.0, right=0.0)
    dv_direct = model.datavector(table)
    rel = np.abs(dv_like - dv_direct)[dv_direct != 0] / np.abs(dv_direct)[dv_direct != 0]
    print(f"B. realization {a}, u = {np.round(u, 2)}: max rel diff likelihood vs direct = {rel.max():.2e}; "
          f"chi2(dv(u), dv(u=0)) = {model.chi2(dv_like, dv_u0):.2f}")
    assert rel.max() < 1e-7          # the printed data vector has 8 significant digits

    # ---- C: shift mode vs Cosmolike's DES_DZ_S ---------------------------------
    # mode 1 = [nbar(z - dz) - nbar(z)] / dz (table shifted by linear interpolation),
    # so u_1 = dz gives the shifted n(z).  Cosmolike's DES_DZ_S shifts its spline
    # of the same table instead, and does not renormalize: the two agree up to
    # interpolation/edge differences (with the DES Y6 n(z), whose last row at
    # z = 2.99 is a pile-up of ~0.3% of the galaxies, the edge effect alone is
    # ~5-10% of the response; see the README).  This is a test of the sign and
    # normalization convention of the modes, not of Cosmolike's interpolation.
    dz = 0.01
    nbar_fine = model.nz_table[:, 1:].T                              # (ntomo, Nz_fine)
    U_shift = np.stack([(np.interp(model.z - dz, model.z, nb, left=0.0, right=0.0) - nb) / dz
                        for nb in nbar_fine])[:, :, None]
    shift_modes_file = os.path.join(data_dir, "_test_shift_modes.txt")
    write_modes_file(shift_modes_file, model.z, U_shift, header="test: mode 1 = [nbar(z-dz)-nbar(z)]/dz")
    shift_dataset = dataset.replace(".dataset", "_shifttest.dataset")
    with open(shift_dataset, "w") as g:
        for line in open(dataset):
            g.write(f"nz_source_modes_file = {os.path.basename(shift_modes_file)}\n"
                    if line.startswith("nz_source_modes_file") else line)
    dv_mode = run_evaluate(base, {"external_nz_modeling": 1, "source_nz_nmodes": 1,
                                  "data_file": os.path.basename(shift_dataset)},
                           {"DES_U_S1": dz}, f"{workdir}/C_mode")
    dv_dz = run_evaluate(base, {"external_nz_modeling": 1, "source_nz_nmodes": 0},
                         {f"DES_DZ_S{i+1}": dz for i in range(ntomo)}, f"{workdir}/C_dz")
    ok = dv_direct != 0
    change_mode = (dv_mode - dv_u0)[ok] / dv_u0[ok]
    change_dz = (dv_dz - dv_u0)[ok] / dv_u0[ok]
    ratio = model.chi2(dv_mode, dv_dz) / model.chi2(dv_dz, dv_u0)
    print(f"C. shift mode u_1 = {dz} vs DES_DZ_S = {dz}: median ratio of the changes = "
          f"{np.median(change_mode / change_dz):.4f}, max |change| = {np.abs(change_dz).max():.3e}, "
          f"chi2(mode, DZ) / chi2(DZ, no shift) = {ratio:.2e}")
    assert abs(np.median(change_mode / change_dz) - 1) < 0.15 and ratio < 0.05
    os.remove(shift_modes_file); os.remove(shift_dataset)
    print("all likelihood tests passed")


if __name__ == "__main__":
    main()
