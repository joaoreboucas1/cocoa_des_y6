# n(z) mode projection ("n(z) PCA") for cocoa_des_y6

DES Y6 does not marginalize over the source redshift distributions with shift
parameters but with *modes*:

    n_i(z) = nbar_i(z) + sum_{j=1}^{M} u_j U_ij(z),      u_j ~ N(0,1) truncated at +-3

(Yin et al. 2025, arXiv:2510.23566, eq. 30; method: Bernstein et al. 2025,
arXiv:2506.00758; same method in CoCoA for Roman: de Souza et al. 2026,
arXiv:2602.09230).  One amplitude `u_j` moves all tomographic bins at once, so
the modes carry the bin-to-bin correlations of the photo-z calibration.
The modes are *Fisher-weighted* principal components of the n(z) realizations:
they are the linear combinations of n(z) variations that change the data
vector the most, in units of its covariance, and each eigenvalue is exactly
the mean chi^2 that would be lost by dropping that mode (eq. 21 of Bernstein).

This folder implements both halves:

| file | what |
|---|---|
| `../likelihood/nz_modes.py` | run-time part: reads the modes file, evaluates the formula above (`NzModes`) |
| `../likelihood/_cosmolike_prototype_base.py` | 12-line hook: `source_nz_nmodes > 0` -> `NzModes` -> `ci.set_source_sample` |
| `fisher_modes.py` | the algebra (pure numpy, equation numbers of Bernstein et al. in the comments) |
| `cocoa_datavector.py` | the cocoa data vector as a function of n(z) only, at the fiducial point of a yaml |
| `compute_jacobian.py` | step 1: J = d(data vector)/d n(z_k) with **derivkit** |
| `compute_modes.py` | step 2: modes + recommended M + the modes file |
| `validate_modes.py` | step 3: check the modes against the real Cosmolike data vector |
| `test_likelihood_modes.py` | end-to-end test of the likelihood path through cobaya |
| `test_fisher_modes.py` | fast unit tests of the algebra (`pytest`) |
| `make_dummy_realizations.py`, `make_gaussian_cov.py` | dummy products to run everything before the DES files exist |

No change to the Cosmolike core or to the C++ interface is needed: the
`external_nz_modeling: 1` hook (`read_redshift_distributions` +
`set_source_sample` at every likelihood call) already exists.

## Quick start (dummy products)

From the `Cocoa/` directory, with cocoa and the project started (`source
start_cocoa.sh; source projects/des_y6/scripts/start_des_y6.sh`), and
`pip install derivkit`:

```bash
P=projects/des_y6
export PYTHONPATH=$P/nz_pca:$PYTHONPATH
python $P/nz_pca/make_dummy_realizations.py --nz $P/data/DESY6_source.nz --out $P/data/dummy_nz_realizations.npz
python $P/nz_pca/make_gaussian_cov.py --yaml $P/EXAMPLE_NZ_MODES_FISHER.yaml \
       --out $P/data/gaussian_xi.cov --out-datavector $P/data/fiducial_xi.modelvector
python $P/nz_pca/compute_jacobian.py --yaml $P/EXAMPLE_NZ_MODES_FISHER.yaml \
       --realizations $P/data/dummy_nz_realizations.npz --out $P/chains/jacobian_dummy.npz
python $P/nz_pca/compute_modes.py --realizations $P/data/dummy_nz_realizations.npz \
       --jacobian $P/chains/jacobian_dummy.npz --out $P/data/dummy_source_modes.txt
python $P/nz_pca/validate_modes.py --yaml $P/EXAMPLE_NZ_MODES_FISHER.yaml \
       --realizations $P/data/dummy_nz_realizations.npz --jacobian $P/chains/jacobian_dummy.npz \
       --modes $P/data/dummy_source_modes.txt --plot $P/chains/dummy_modes_validation.png
python $P/nz_pca/test_likelihood_modes.py --yaml $P/EXAMPLE_NZ_MODES_FISHER.yaml \
       --realizations $P/data/dummy_nz_realizations.npz --modes $P/data/dummy_source_modes.txt
cobaya-run $P/EXAMPLE_NZ_MODES_MCMC.yaml
```

`EXAMPLE_NZ_MODES_FISHER.yaml` points at `DESY6_nzpca_test.dataset`, whose
covariance and data vector (`gaussian_xi.cov`, `fiducial_xi.modelvector`) do not
exist until step 2; `make_gaussian_cov.py` therefore initializes the likelihood
with `--bootstrap-dataset DESY6.dataset` (the repo's dummy files) and writes
both.  Only the covariance and the mask matter for the Jacobian.


## Using it in a chain

1. `.dataset` file: add `nz_source_modes_file = <modes file>` (next to `nz_source_file`).
2. likelihood yaml (`cosmic_shear.yaml` or the run yaml): `external_nz_modeling: 1`,
   `source_nz_nmodes: M` (M <= number of modes in the file; DES Y6 used 7).
3. params: `DES_U_S1 ... DES_U_SM` with prior `norm(0,1)` in `[-3, 3]`, and
   `DES_DZ_S* : {value: 0}` (DES Y6 does not use shifts together with modes;
   both can be on at the same time if you want the "m + Delta z" robustness
   variant).  `EXAMPLE_NZ_MODES_MCMC.yaml` is a complete example.

The likelihood evaluates `n_i(z) = nbar_i(z) + sum_j u_j U_ij(z)` on the z grid
of the `.nz` file (modes on a coarser grid are linearly interpolated, which is
the "triangular kernel" rebinning DES uses) and passes the table to Cosmolike,
which normalizes each bin to unit integral (the modes have zero integral up to
the realizations' own normalization scatter, so this is a no-op in practice)
and does not clip negative values, exactly as DES's CosmoSIS pipeline.

## Modes file format

Text, one row per (bin, z), bin-major, columns `bin z U_1(z) ... U_Mmax(z)`
(written by `nz_modes.write_modes_file`).  