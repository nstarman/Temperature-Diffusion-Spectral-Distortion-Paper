"""Simulate the P2LS data on a grid of spll, sprp."""

import sys

import asdf
import astropy.units as u
import numpy as np
from showyourwork.paths import user as user_paths

paths = user_paths()

sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.correlation.core import calculate_correlation_z1z2
from scripts.cosmo import cosmo

##############################################################################

try:
    snkmkp: dict[str, int] = snakemake.params  # type: ignore[name-defined]
except NameError:
    snkmkp = {"seed": 78}

rng = np.random.default_rng(snkmkp["seed"])

# -----------------------------------------------------------------
# Load

with asdf.open(paths.data / "variables.asdf", lazy_load=False, copy_arrays=True) as vs:
    z0 = vs["z0"]
    z_eq = vs["z_eq"]
    pivot_scale = vs["pivot_scale"]
    ns = vs["ns"]
    z_reco = vs["z_recombination"]
    Leq = vs["Leq"]
    r0 = vs["r0"]

with asdf.open(paths.data / "P2LS.asdf", lazy_load=False, copy_arrays=True) as af:
    kd_2ls = af["kd_2ls"]

with (paths.data / "p2ls_samples.npy").open("rb") as f:
    samples = np.load(f)

# -----------------------------------------------------------------

drs = np.geomspace(1e-3, 3e3, num=2_500) * u.Mpc
ks = np.geomspace(1e-4, 1_000, num=200_000) / u.Mpc
chis = np.linspace(0, np.pi / 20, num=500) * u.rad
ys = calculate_correlation_z1z2(
    z_reco,
    z0,
    chis,
    cosmo=cosmo,
    samples=samples,
    ks=ks,
    drs=drs,
    Leq=Leq,
    r0=r0,
    ns=ns,
    z_eq=z_eq,
    kdamping=kd_2ls,
    pivot_scale=pivot_scale,
    rng=rng,
    n_chi_steps=50,
)

with (paths.data / "C_EL_corr.npy").open("wb") as f:
    np.save(f, (chis, ys))
