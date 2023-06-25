"""CAMB cosmology constants.

From the :mod:`cosmology.api`, the list of required constants is:

- c: Speed of light in km s-1.
- G: Gravitational constant G in pc km2 s-2 Msol-1.
"""

__all__ = ["c", "G"]

import astropy.units as u
from cosmology.compat.classy import constants

c: u.Quantity = u.Quantity(constants.c, u.km / u.s)
G: u.Quantity = u.Quantity(constants.G, u.pc * u.km**2 / u.s**2 / u.Msun)
