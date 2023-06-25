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

with asdf.open(paths.data / "variables.asdf", lazy_load=False) as vs:
    As = vs["As"]

chis_EL, C_EL_corr = np.load(paths.data / "C_EL_corr.npy")

i = np.argmin(np.abs(chis_EL * 180 / np.pi - 1))
C_EL_one = C_EL_corr[i]

var_YT2_pixie = 2e-6 * 3e-10 / np.sqrt(3e5)

with (paths.output / "sn_pixie.txt").open("w") as f:
    f.write(rf"{scientific_format(C_EL_one * As ** 2 / var_YT2_pixie)}")
