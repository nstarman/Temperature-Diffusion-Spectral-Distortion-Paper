"""Simulate the P2LS data on a grid of spll, sprp."""
# ruff: noqa: ERA001

import sys

import asdf
import astropy.units as u
import numpy as np
from showyourwork.paths import user as user_paths

paths = user_paths()

sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.cosmo import cosmo

##############################################################################

# with asdf.open(paths.data / "variables.asdf", lazy_load=False) as vs:
#     As = vs["As"]
# chis_EL, C_EL_corr = np.load(paths.data / "C_EL_corr.npy")
# C_EL_zero = C_EL_corr[0]
# var_YT2_S4 = 3e-6 * 3e-10 / np.sqrt(3e5 * 60**2)

# with (paths.output / "sn_act.txt").open("w") as f:
#     f.write(rf"{C_EL_zero * As ** 2 / var_YT2_S4:0.0f}")

# Load data
npzfile = np.load(paths.data / "cls.npy")

with asdf.open(paths.data / "variables.asdf", lazy_load=False) as vs:
    As = vs["As"]

# Get ls and Cls
lmin, lmax = 500, 1500
ls = npzfile["ls"][lmin : lmax + 1]
cls_EL = As**2 * npzfile["cls_EL"][lmin : lmax + 1]
cls_LL = As**2 * npzfile["cls_LL"][lmin : lmax + 1]

# Other terms
Npix = 4.8e7
fsky = 0.33
delta_T = 10 * u.uK
delta_Y = (1.6 * delta_T / cosmo.T_cmb0).decompose()
winv = 4 * np.pi * delta_Y**2 / Npix
theta_fwhm = 1.4 * u.arcmin  # ACT at 90 GHz
sigma_b = 1.2e-4 * theta_fwhm.to_value(u.arcmin)
bigWb = np.exp(-(ls**2) * sigma_b**2 / 2)

# Calculate
sn_ls = np.sqrt((2 * ls + 1) * fsky / (cls_LL * winv)) * cls_EL * bigWb
sn = np.sum(sn_ls).decompose()

with (paths.output / "sn_act.txt").open("w") as f:
    f.write(f"{sn:.0f}")
