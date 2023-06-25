"""The Cosmology API compatability wrapper for CAMB."""

from __future__ import annotations

__all__: list[str] = []

from dataclasses import dataclass
from typing import TYPE_CHECKING, overload

import astropy.units as u
import numpy as np
from cosmology.compat.classy import (
    StandardCosmologyWrapper as CLASSYStandardCosmologyWrapper,
)
from cosmology.compat.classy import constants as c

from . import constants

if TYPE_CHECKING:
    from cosmology.api import CosmologyConstantsNamespace
    from cosmology.compat.classy._core import Array, InputT


@dataclass(frozen=True)
class StandardCosmologyWrapper(CLASSYStandardCosmologyWrapper):  # type: ignore[misc]
    """FLRW Cosmology API wrapper for CAMB cosmologies."""

    @property
    def constants(self) -> CosmologyConstantsNamespace:
        """Cosmology constants."""
        return constants

    # ----------------------------------------------
    # CriticalDensity

    @property
    def critical_density0(self) -> u.Quantity:
        """Critical density at z = 0 in Msol Mpc-3."""
        return u.Quantity(3e6 * self.H0**2 / (8 * np.pi * c.G), u.Msun / u.Mpc**3)

    def critical_density(self, z: InputT, /) -> u.Quantity:
        """Redshift-dependent critical density in Msol Mpc-3."""
        return u.Quantity(3e6 * self.H(z) ** 2 / (8 * np.pi * c.G), u.Msun / u.Mpc**3)

    # ----------------------------------------------
    # HubbleParameter

    @property
    def H0(self) -> u.Quantity:
        """Hubble constant at z=0 in km s-1 Mpc-1."""
        return u.Quantity(c.c * self.cosmo.Hubble(0), u.km / u.s / u.Mpc)

    @property
    def hubble_distance(self) -> u.Quantity:
        """Hubble distance in Mpc."""
        return u.Quantity(1 / self.cosmo.Hubble(0), u.Mpc)

    @property
    def hubble_time(self) -> u.Quantity:
        """Hubble time in Gyr."""
        return (1 / self.H0).to(u.Gyr)

    def H(self, z: InputT, /) -> u.Quantity:
        """Hubble function :math:`H(z)` in km s-1 Mpc-1."""  # noqa: D402
        return u.Quantity(c.c * self._cosmo_fn["Hubble"](z), u.km / u.s / u.Mpc)

    def H_over_H0(self, z: InputT, /) -> Array:
        """Standardised Hubble function :math:`E(z) = H(z)/H_0`."""
        return self._cosmo_fn["Hubble"](z) / self.cosmo.Hubble(0)

    # ----------------------------------------------
    # Temperature

    @property
    def T_cmb0(self) -> u.Quantity:
        """Temperature of the CMB at z=0."""
        return u.Quantity(self.cosmo.T_cmb(), u.K)

    def T_cmb(self, z: InputT, /) -> u.Quantity:
        """Temperature of the CMB at redshift ``z``."""
        return self.T_cmb0 * (z + 1)

    # ----------------------------------------------
    # Comoving distance

    @overload
    def comoving_distance(self, z: InputT, /) -> u.Quantity:
        ...

    @overload
    def comoving_distance(self, z1: InputT, z2: InputT, /) -> u.Quantity:
        ...

    def comoving_distance(self, z1: InputT, z2: InputT | None = None, /) -> u.Quantity:
        r"""Comoving line-of-sight distance :math:`d_c(z)` in Mpc.

        The comoving distance along the line-of-sight between two objects
        remains constant with time for objects in the Hubble flow.

        Parameters
        ----------
        z : Array, positional-only
        z1, z2 : Array, positional-only
            Input redshifts. If one argument ``z`` is given, the distance
            :math:`d_c(0, z)` is returned. If two arguments ``z1, z2`` are
            given, the distance :math:`d_c(z_1, z_2)` is returned.

        Returns
        -------
        Array
            The comoving distance :math:`d_c` in Mpc.
        """
        z1, z2 = (0, z1) if z2 is None else (z1, z2)
        return (
            self._cosmo_fn["comoving_distance"](z2)
            - self._cosmo_fn["comoving_distance"](z1)
            << u.Mpc
        )

    def _comoving_volume_flat(self, z: InputT, /) -> u.Quantity:
        return 4.0 / 3.0 * np.pi * self.comoving_distance(z) ** 3

    def _comoving_volume_positive(self, z: InputT, /) -> u.Quantity:
        dh = self.hubble_distance
        x = self.transverse_comoving_distance(z) / dh
        term1 = 4.0 * np.pi * dh**3 / (2.0 * self.Omega_k0)
        term2 = x * np.sqrt(1 + self.Omega_k0 * (x) ** 2)
        term3 = np.sqrt(np.abs(self.Omega_k0)) * x

        return term1 * (
            term2 - 1.0 / np.sqrt(np.abs(self.Omega_k0)) * np.arcsinh(term3)
        )

    def _comoving_volume_negative(self, z: InputT, /) -> u.Quantity:
        dh = self.hubble_distance
        x = self.transverse_comoving_distance(z) / dh
        term1 = 4.0 * np.pi * dh**3 / (2.0 * self.Omega_k0)
        term2 = x * np.sqrt(1 + self.Omega_k0 * (x) ** 2)
        term3 = np.sqrt(np.abs(self.Omega_k0)) * x
        return term1 * (term2 - 1.0 / np.sqrt(np.abs(self.Omega_k0)) * np.arcsin(term3))

    @overload
    def comoving_volume(self, z: InputT, /) -> u.Quantity:
        ...

    @overload
    def comoving_volume(self, z1: InputT, z2: InputT, /) -> u.Quantity:
        ...

    def comoving_volume(self, z1: InputT, z2: InputT | None = None, /) -> u.Quantity:
        r"""Comoving volume in cubic Mpc.

        This is the volume of the universe encompassed by redshifts less than
        ``z``. For the case of :math:`\Omega_k = 0` it is a sphere of radius
        `comoving_distance` but it is less intuitive if :math:`\Omega_k` is not.
        """
        if z2 is not None:
            raise NotImplementedError

        if self.Omega_k0 == 0:
            cv = self._comoving_volume_flat(z1)
        elif self.Omega_k0 > 0:
            cv = self._comoving_volume_positive(z1)
        else:
            cv = self._comoving_volume_negative(z1)
        return cv << u.Mpc**3

    def differential_comoving_volume(self, z: InputT, /) -> Array:
        r"""Differential comoving volume in cubic Mpc per steradian.

        If :math:`V_c` is the comoving volume of a redshift slice with solid
        angle :math:`\Omega`, this function ...

        .. math::

            \mathtt{dvc(z)}
            = \frac{1}{d_H^3} \, \frac{dV_c}{d\Omega \, dz}
            = \frac{x_M^2(z)}{E(z)}
            = \frac{\mathtt{xm(z)^2}}{\mathtt{ef(z)}} \;.

        """
        return (
            self.transverse_comoving_distance(z) / self.hubble_distance
        ) ** 2 / self.H_over_H0(z) << u.Mpc**3 / u.sr

    # ----------------------------------------------
    # Angular diameter

    @overload
    def angular_diameter_distance(self, z: InputT, /) -> u.Quantity:
        ...

    @overload
    def angular_diameter_distance(self, z1: InputT, z2: InputT, /) -> u.Quantity:
        ...

    def angular_diameter_distance(
        self, z1: InputT, z2: InputT | None = None, /
    ) -> u.Quantity:
        """Angular diameter distance :math:`d_A` in Mpc.

        This gives the proper (sometimes called 'physical') transverse distance
        corresponding to an angle of 1 radian for an object at redshift ``z``
        ([1]_, [2]_, [3]_).

        Parameters
        ----------
        z : Array, positional-only
        z1, z2 : Array, positional-only
            Input redshifts. If one argument ``z`` is given, the distance
            :math:`d_A(0, z)` is returned. If two arguments ``z1, z2`` are
            given, the distance :math:`d_A(z_1, z_2)` is returned.

        Returns
        -------
        Array
            The angular diameter distance :math:`d_A` in Mpc.

        References
        ----------
        .. [1] Weinberg, 1972, pp 420-424; Weedman, 1986, pp 421-424.
        .. [2] Weedman, D. (1986). Quasar astronomy, pp 65-67.
        .. [3] Peebles, P. (1993). Principles of Physical Cosmology, pp 325-327.
        """
        if z2 is not None:
            raise NotImplementedError
        return u.Quantity(self._cosmo_fn["angular_distance"](z1), u.Mpc)

    # ----------------------------------------------
    # Luminosity distance

    @overload
    def luminosity_distance(self, z: InputT, /) -> u.Quantity:
        ...

    @overload
    def luminosity_distance(self, z1: InputT, z2: InputT, /) -> u.Quantity:
        ...

    def luminosity_distance(
        self, z1: InputT, z2: InputT | None = None, /
    ) -> u.Quantity:
        """Redshift-dependent luminosity distance :math:`d_L` in Mpc.

        This is the distance to use when converting between the bolometric flux
        from an object at redshift ``z`` and its bolometric luminosity [1]_.

        Parameters
        ----------
        z : Array, positional-only
        z1, z2 : Array, positional-only
            Input redshifts. If one argument ``z`` is given, the distance
            :math:`d_L(0, z)` is returned. If two arguments ``z1, z2`` are
            given, the distance :math:`d_L(z_1, z_2)` is returned.

        Returns
        -------
        Array
            The luminosity distance :math:`d_L` in Mpc.

        References
        ----------
        .. [1] Weinberg, 1972, pp 420-424; Weedman, 1986, pp 60-62.
        """
        if z2 is not None:
            raise NotImplementedError
        return u.Quantity(self._cosmo_fn["luminosity_distance"](z1), u.Mpc)
