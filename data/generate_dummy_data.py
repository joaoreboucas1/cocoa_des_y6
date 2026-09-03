import numpy as np

NUM_ANG_BINS = 20
NUM_SRC_BINS = 4
NUM_LENS_BINS = 6

DV_LEN = NUM_SRC_BINS*(NUM_SRC_BINS+1)*NUM_ANG_BINS + NUM_SRC_BINS*NUM_LENS_BINS*NUM_ANG_BINS + NUM_LENS_BINS*NUM_ANG_BINS

mask = np.ones(DV_LEN)
indices = np.arange(DV_LEN)
np.savetxt("ones.mask", np.column_stack((indices, mask)), fmt="%d")
np.savetxt("dummy.modelvector", np.column_stack((indices, mask)), fmt="%d %.8e")
np.savetxt("dummy.cov", np.column_stack((indices, indices, mask)), fmt="%d")

