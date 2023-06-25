"""Setup."""

import sys

import asdf
import astropy.cosmology.units as cu
import astropy.units as u
from showyourwork.paths import user as user_paths

paths = user_paths()

sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.cosmo import cosmo, filename, params
from scripts.src.distances import Leq, s_of_z, z_recombination

##############################################################################
# Using the "Combined" column from Planck18 Cosmological Parameters Table 1

af = asdf.AsdfFile()

af.tree["filename"] = filename

af.tree["As"] = float(params["A_s"]) * u.one
af.tree["ns"] = float(params["n_s"]) * u.one
af.tree["pivot_scale"] = 0.05 / u.Mpc

af.tree["zmax"] = zmax = 5500 * cu.redshift
af.tree["z0"] = z0 = 100 * cu.redshift
af.tree["z_eq"] = z_eq = cosmo.Omega_m0 / cosmo.cosmo.Omega_r() - 1 << cu.redshift

af.tree["z_recombination"] = z_reco = z_recombination(cosmo) << cu.redshift

af.tree["Leq"] = Leq(cosmo, z_eq=z_eq) << u.Mpc

af.tree["r0"] = cosmo.comoving_distance(z0) << u.Mpc

# S
af.tree["s0"] = s_of_z(z0, z0=z0, z_eq=z_eq) << u.one
af.tree["smax"] = s_of_z(zmax, z0=z0, z_eq=z_eq) << u.one
af.tree["s_recombination"] = s_of_z(z_reco, z0=z0, z_eq=z_eq) << u.one

af.write_to(paths.data / "variables.asdf")
