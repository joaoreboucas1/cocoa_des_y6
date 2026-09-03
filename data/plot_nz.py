import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ERROR: .nz file not provided")
        print(f"Usage: python3 {sys.argv[0]} nzfile1 [nzfile2 ...]")
        exit(1)

    nzfiles = sys.argv[1:]
    colors = mcolors.TABLEAU_COLORS

    for nzfile, color in zip(nzfiles, colors):
        if not os.path.exists(nzfile):
            print(f"ERROR: could not find .nz file {nzfile}")
            print(f"Usage: python3 {sys.argv[0]} nzfile1 [nzfile2 ...]")
            exit(1)

        data = np.loadtxt(nzfile, unpack=True)
        z = data[0]
        nz = data[1:]
        num_bins = len(nz)

        for i, nz_i in enumerate(nz):
            plt.plot(z, nz_i, color=color, label=nzfile if i == 0 else "")
    plt.legend()
    plt.show()