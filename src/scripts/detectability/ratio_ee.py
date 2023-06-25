"""Simulate the P2LS data on a grid of spll, sprp."""

import sys

import numpy as np
from showyourwork.paths import user as user_paths

paths = user_paths()

sys.path.append(paths.scripts.parent.as_posix())
# isort: split

##############################################################################

# Load data
chis_EE, C_EE_corr = np.load(paths.data / "C_EE_corr.npy")

# C_EL at 0 degrees
C_EE_zero = C_EE_corr[0]

# C_EL at 1 degrees
i = np.argmin(np.abs(chis_EE * 180 / np.pi - 1))
C_EE_one = C_EE_corr[i]

with (paths.output / "C_EE_ratio.txt").open("w") as f:
    f.write(f"{C_EE_zero / C_EE_one:.0f}")
