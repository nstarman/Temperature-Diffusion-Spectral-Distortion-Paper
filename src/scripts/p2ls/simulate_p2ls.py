"""Simulate the P2LS data on a grid of spll, sprp."""

import sys

import asdf
import astropy.cosmology.units as cu
import astropy.units as u
import numpy as np
from showyourwork.paths import user as user_paths

paths = user_paths()

sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.cosmo import cosmo
from scripts.src.distances import z_of_s
from scripts.src.prob_2ls import ComputePspllSprp, compute_on_grid

##############################################################################

try:
    snkmkp: dict[str, int] = snakemake.params  # type: ignore[name-defined]
except NameError:
    snkmkp = {"n_sprp": 15, "n_z_center": 1_000, "n_z_lr": 1_000}

u.add_enabled_units(cu)

# -----------------------------------------------------------------
# Run

# Load the relevant variables
with asdf.open(paths.data / "variables.asdf", lazy_load=False, copy_arrays=True) as vs:
    z0 = vs["z0"]
    z_eq = vs["z_eq"]
    s_reco = vs["s_recombination"]
    smax = vs["smax"]
    zmax = vs["zmax"]

spll = np.linspace(0.5, s_reco + 0.15, num=300) * u.one
sprp = np.geomspace(1e-7, smax, num=400) * u.one
Spll, Sprp = np.meshgrid(spll, sprp, indexing="ij")

Pcomputer = ComputePspllSprp.from_CLASS(cosmo, z_eq=z_eq, z_bounds=(zmax, z0))
Pspllsprp, correction = compute_on_grid(Pcomputer, Spll, Sprp, **snkmkp)

##############################################################################
# Save data

af = asdf.AsdfFile()
af.tree["spll"] = spll
af.tree["sprp"] = sprp
af.tree["Pspllsprp"] = Pspllsprp
af.tree["correction"] = correction

af.tree["meta"] = dict(snkmkp)

# Extras used in other scripts
af.tree["z_2ls"] = (
    z_of_s(spll[np.argmax(Pspllsprp[:, 0])], z0=z0, z_eq=z_eq) << cu.redshift
)
af.tree["kd_2ls"] = 1 / 8.8 / u.Mpc  # Baumann book

af.write_to(paths.data / "P2LS.asdf")
