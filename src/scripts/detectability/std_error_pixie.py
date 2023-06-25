"""Simulate the P2LS data on a grid of spll, sprp."""

import sys

import astropy.units as u
import numpy as np
from showyourwork.paths import user as user_paths

paths = user_paths()

sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.cosmo import cosmo
from scripts.utils import scientific_format

##############################################################################

Ntheta_pixie = 3e5
error_in_y_pixie = 2e-6

chis_LL, C_LL_corr = np.load(paths.data / "C_LL_corr.npy")
i = np.argmin(np.abs(chis_LL * 180 / np.pi - 1))
C_EL_one = C_LL_corr[i]
expectation_deltaT22 = C_EL_one * cosmo.T_cmb0**4

result = error_in_y_pixie * np.sqrt(expectation_deltaT22 / Ntheta_pixie)

with (paths.output / "std_error_pixie.txt").open("w") as f:
    f.write(scientific_format(result.to_value(u.K**2), decimals=1))
