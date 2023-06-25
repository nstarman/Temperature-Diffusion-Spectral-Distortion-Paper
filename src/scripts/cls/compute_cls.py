"""Simulate the P2LS data on a grid of spll, sprp."""

import flt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from showyourwork.paths import user as user_paths

paths = user_paths()

##############################################################################


closed = True
thetas = flt.theta(int(1e5), closed=closed)


chis_EE, C_EE_corr = np.load(paths.data / "C_EE_corr.npy")
sel = ~np.isnan(C_EE_corr)
C_EE_padded = InterpolatedUnivariateSpline(
    chis_EE[sel], C_EE_corr[sel], k=3, ext="zeros"
)(thetas)
cls_EE = flt.dlt(C_EE_padded, closed=closed)


chis_EL, C_EL_corr = np.load(paths.data / "C_EL_corr.npy")
sel = ~np.isnan(C_EL_corr)
C_EL_padded = InterpolatedUnivariateSpline(
    chis_EL[sel], C_EL_corr[sel], k=3, ext="zeros"
)(thetas)
cls_EL = flt.dlt(C_EL_padded, closed=closed)


chis_LL, C_LL_corr = np.load(paths.data / "C_LL_corr.npy")
sel = ~np.isnan(C_LL_corr)
C_LL_padded = InterpolatedUnivariateSpline(
    chis_LL[sel], C_LL_corr[sel], k=3, ext="zeros"
)(thetas)
cls_LL = flt.dlt(C_LL_padded, closed=closed)


##############################################################################
# Save data

ls = np.arange(0, len(cls_EE) + 1)

with (paths.data / "cls.npy").open("wb") as f:
    np.savez(f, ls=ls, thetas=thetas, cls_EE=cls_EE, cls_EL=cls_EL, cls_LL=cls_LL)
