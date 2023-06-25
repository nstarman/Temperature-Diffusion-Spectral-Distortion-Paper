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

chis_EE, C_EE_corr = np.load(paths.data / "C_EE_corr.npy")

# C_EE at 0 degrees
C_EE_zero = C_EE_corr[0]

# Save
with (paths.output / "C_EE_zero.txt").open("w") as f:
    f.write(
        rf"{scientific_format(C_EE_zero)} A_s^2 "
        rf"\simeq {scientific_format(C_EE_zero * As ** 2)}"
    )
