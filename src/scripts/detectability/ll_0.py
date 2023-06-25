"""Simulate the P2LS data on a grid of spll, sprp."""

import sys

import asdf
import numpy as np
from showyourwork.paths import user as user_paths

paths = user_paths()

sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.utils import scientific_format

##############################################################################

# Load data
with asdf.open(paths.data / "variables.asdf", lazy_load=False) as vs:
    As = vs["As"]

chis_LL, C_LL_corr = np.load(paths.data / "C_LL_corr.npy")

# C_LL at 0 degrees
C_LL_zero = C_LL_corr[0]

# Save
with (paths.output / "C_LL_zero.txt").open("w") as f:
    f.write(rf"{C_LL_zero:.0f} A_s^2 \simeq {scientific_format(C_LL_zero * As ** 2)}")
