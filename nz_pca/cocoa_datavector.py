"""
The cocoa/Cosmolike data vector as a function of the source n(z) only.

Everything else (cosmology, IA, shear calibration, photo-z shifts, scale
cuts) is frozen at the fiducial point of a cobaya yaml file, exactly as in a
likelihood evaluation.  This is the function whose derivative with respect to
n(z) gives the Jacobian J of Bernstein et al. 2025 (eq. 10).

Usage
-----
    model = CocoaDataVector("EXAMPLE_NZ_FISHER.yaml")
    dv0   = model.datavector(model.nz_table)          # fiducial
    dv1   = model.datavector(perturbed_table)         # any (Nz, 1+Ntomo) table
    F     = J.T @ model.invcov @ J

The yaml must be a normal cocoa yaml (likelihood/theory/params blocks) whose
`sampler: evaluate: override:` block fixes ALL sampled parameters: that is the
fiducial point.  The likelihood must be a des_y6 one (it may have
`external_nz_modeling: 0`; only `set_source_sample` is used here).
"""
import numpy as np
from cobaya.yaml import yaml_load_file
from cobaya.model import get_model

import cosmolike_des_y6_interface as ci


class CocoaDataVector:

    def __init__(self, yaml_file, likelihood_options=None):
        """likelihood_options: dict overriding options of the likelihood block of
        the yaml (e.g. {"data_file": "DESY6.dataset"} to use another .dataset)."""
        info = yaml_load_file(yaml_file)
        if likelihood_options:
            like_name = list(info["likelihood"])[0]
            info["likelihood"][like_name].update(likelihood_options)
        self.model = get_model(info)
        self.like = list(self.model.likelihood.values())[0]

        # fiducial point = the evaluate override; all sampled params must be there
        fiducial = dict(info["sampler"]["evaluate"]["override"])
        sampled = self.model.parameterization.sampled_params()
        missing = [p for p in sampled if p not in fiducial]
        if missing:
            raise ValueError(f"{yaml_file}: sampler.evaluate.override must fix all "
                             f"sampled parameters; missing {missing}")
        point = {p: fiducial[p] for p in sampled}

        # one likelihood evaluation: sets cosmology + nuisance in Cosmolike
        self.logpost = self.model.logposterior(point)
        self.fiducial = dict(self.model.parameterization.constant_params(), **point)

        self.nz_table = np.loadtxt(self.like.source_file)     # (Nz, 1+Ntomo)
        self.z = self.nz_table[:, 0]
        self.ntomo = self.nz_table.shape[1] - 1
        self.dz_fiducial = [self.fiducial.get(f"DES_DZ_S{i+1}", 0.0) for i in range(self.ntomo)]
        self.invcov = np.array(ci.get_inv_cov_masked())
        self.mask = np.array(ci.get_mask())
        self.dv_fiducial = self.datavector(self.nz_table)

    def datavector(self, nz_table):
        """Full (masked) data vector for the given (Nz, 1+Ntomo) n(z) table."""
        # NOTE: the C++ side (carma) takes ownership of the numpy buffer it is
        # given ("steals" it), so always pass a fresh copy, never an array that
        # is used again afterwards.  Same rule as `self.source_nz.copy()` in
        # _cosmolike_prototype_base.py.
        ci.set_source_sample(np.array(nz_table, dtype="float64", order="F"))
        return np.array(ci.compute_data_vector_masked(), dtype="float64")

    def datavector_with_shift(self, delta_z):
        """Data vector with the fiducial n(z) shifted by n_i(z - delta_z_i)
        (Cosmolike's DES_DZ_S model). The shifts are reset to fiducial afterwards."""
        ci.set_nuisance_shear_photoz(bias=list(np.asarray(delta_z, dtype="float64")))
        dv = self.datavector(self.nz_table)
        ci.set_nuisance_shear_photoz(bias=list(self.dz_fiducial))
        return dv

    def chi2(self, dv_a, dv_b):
        d = dv_a - dv_b
        return float(d @ self.invcov @ d)
