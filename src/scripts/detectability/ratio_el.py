"""Simulate the P2LS data on a grid of spll, sprp."""

import numpy as np
from showyourwork.paths import user as user_paths

paths = user_paths()

##############################################################################

# Load data
chis_EL, C_EL_corr = np.load(paths.data / "C_EL_corr.npy")

# C_EL at 0 degrees
C_EL_zero = C_EL_corr[0]

# C_EL at 1 degrees
i = np.argmin(np.abs(chis_EL * 180 / np.pi - 1))
C_EL_one = C_EL_corr[i]

with (paths.output / "C_EL_ratio.txt").open("w") as f:
    f.write(f"{C_EL_zero / C_EL_one:.0f}")
