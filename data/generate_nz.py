import os
import numpy as np

cosmosis_path  = "/Users/joao/cosmo/cocoa/Cocoa/projects/y6_code_comparison/cosmosis/lcdm_datavector_run/"
nz_source_path = f"{cosmosis_path}/nz_source/"
nz_lens_path   = f"{cosmosis_path}/nz_lens/"
NUM_SOURCE_BINS = 4
NUM_LENS_BINS = 6

redshifts  = np.loadtxt(f"{nz_source_path}/z.txt")
nzs_source = [np.loadtxt(f"{nz_source_path}/bin_{i}.txt") for i in range(1, NUM_SOURCE_BINS+1)]
nzs_lens   = [np.loadtxt(f"{nz_lens_path}/bin_{i}.txt")   for i in range(1, NUM_LENS_BINS+1)]

nzhisto_lens = np.vstack([redshifts] + nzs_lens).T
nzhisto_source = np.vstack([redshifts] + nzs_source).T
np.savetxt("DESY6_lens.nz", nzhisto_lens)
np.savetxt("DESY6_source.nz", nzhisto_source)