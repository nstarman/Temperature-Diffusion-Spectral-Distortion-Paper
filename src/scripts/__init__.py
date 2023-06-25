"""Paper codes."""

import astropy.cosmology.units as cu
import astropy.units as u

u.add_enabled_units(cu)
u.add_enabled_equivalencies(u.dimensionless_angles())
