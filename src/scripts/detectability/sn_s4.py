"""Simulate the P2LS data on a grid of spll, sprp."""

import sys

import asdf
import numpy as np
from showyourwork.paths import user as user_paths

paths = user_paths()

sys.path.append(paths.scripts.parent.as_posix())
# isort: split


##############################################################################

with asdf.open(paths.data / "variables.asdf", lazy_load=False) as vs:
    As = vs["As"]

chis_EL, C_EL_corr = np.load(paths.data / "C_EL_corr.npy")

C_EL_zero = C_EL_corr[0]

var_YT2_S4 = 3e-7 * 3e-10 / np.sqrt(3e5 * 60**2)

with (paths.output / "sn_s4.txt").open("w") as f:
    f.write(rf"{C_EL_zero * As ** 2 / var_YT2_S4:.0f}")
