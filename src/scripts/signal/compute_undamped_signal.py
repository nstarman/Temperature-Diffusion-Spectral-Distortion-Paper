"""Simulate the P2LS data on a grid of spll, sprp."""

import sys
from typing import Annotated as Ann
from typing import cast

import asdf
import astropy.units as u
import numpy as np
from interpolated_coordinates.utils import InterpolatedUnivariateSplinewithUnits
from showyourwork.paths import user as user_paths
from typing_extensions import Doc

paths = user_paths()

sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.cosmo import cosmo
from scripts.src.cf import correlation_function
from scripts.src.distances import r_distance_between_s
from scripts.src.typing import NDAf
from scripts.utils import scientific_format

##############################################################################

snkmkp: dict[str, int]
try:
    snkmkp = snakemake.params  # type: ignore[name-defined]
except NameError:
    snkmkp = {"seed": 99}

rng = np.random.default_rng(snkmkp["seed"])

# -----------------------------------------------------------------
# Load

with asdf.open(paths.data / "variables.asdf", lazy_load=False, copy_arrays=True) as vs:
    z_eq = vs["z_eq"]
    z_recombination = vs["z_recombination"]
    As = vs["As"]
    ns = vs["ns"]
    pivot_scale = vs["pivot_scale"]
    Leq = vs["Leq"]

with asdf.open(paths.data / "P2LS.asdf", lazy_load=False, copy_arrays=True) as af:
    spll = af["spll"]
    sprp = af["sprp"]
    Pspllsprp = af["Pspllsprp"]

with (paths.data / "p2ls_samples.npy").open("rb") as f:
    samples = np.load(f)

# -----------------------------------------------------------------

drs = np.geomspace(1e-3, 3e3, num=2_500) * u.Mpc
ks = np.geomspace(1e-4, 1_000, num=200_000) / u.Mpc

# Calculate the correlation function (divded by A_s) [K^2]
xiEE = correlation_function(  # (R,)
    cosmo,
    ks,
    drs,
    z1=z_recombination,
    z2=z_recombination,
    pivot_scale=pivot_scale,
    ns=ns,
    z_eq=z_eq,
    kdamping=np.inf / u.Mpc,  # No damping
    ius_kw={"k": 3, "ext": 2},
)
# Splining for later integration
xiEEspl = InterpolatedUnivariateSplinewithUnits(drs, xiEE, k=3, ext=2)

# Getting random indices for Monte Carlo integration
idx1, idx2 = rng.integers(0, len(samples), size=(2, 10_000_000))
# Cleaning the indices
repeats = idx1 == idx2
idx1 = idx1[~repeats]
idx2 = idx2[~repeats]


# Integral[xi(s2 - s1)] ds1 ds2
def weighted_xi_integral(
    xispl: Ann[InterpolatedUnivariateSplinewithUnits, Doc("Spline of xi")],
    samples1: Ann[NDAf, Doc("(N, 3) scattering location sample 1")],
    samples2: Ann[NDAf, Doc("(N, 3) scattering location sample 2")],
) -> NDAf:
    """Correlation integral."""
    rdist = r_distance_between_s(samples1, samples2, Leq=Leq)
    return cast("NDAf", np.sum(xispl(rdist))) / len(rdist)


rmin = 1e-3 * u.Mpc
xiEE0 = xiEEspl(rmin) - rmin * xiEEspl.derivative()(rmin)
integral = weighted_xi_integral(xiEEspl, samples[idx1], samples[idx2])

signal = As * (xiEE0 - integral) / cosmo.T_cmb0**2
signal = signal.to_value(u.one)

##############################################################################

with (paths.output / "undamped_signal.txt").open("w") as f:
    f.write(scientific_format(signal, decimals=1))
