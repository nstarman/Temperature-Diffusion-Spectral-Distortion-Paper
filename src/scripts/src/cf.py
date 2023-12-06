"""Correlation Function."""

from __future__ import annotations

__all__: list[str] = []

from typing import TYPE_CHECKING, Any, cast
from typing import Annotated as Ann

import numpy as np
from interpolated_coordinates.utils import InterpolatedUnivariateSplinewithUnits

if TYPE_CHECKING:
    from cosmology.api import StandardCosmology
    from typing_extensions import Doc

    from .typing import NDAf


def power_spectrum(
    kmag: Ann[NDAf, Doc("Magnitude of k [Mpc-1]")],
    /,
    pivot_scale: Ann[NDAf | float, Doc("Pivot scale [Mpc-1]")],
    ns: Ann[NDAf | float, Doc("Scalar spectral index")],
) -> Ann[NDAf, Doc("Power spectrum [Mpc3]")]:
    """Compute simple power spectrum (divided by :math:`A_s`)."""
    return cast(
        "NDAf", (np.power(kmag / pivot_scale, ns - 1) / (4 * np.pi * kmag**3))
    )


def _ratio(cosmo: StandardCosmology[NDAf, Any], z: NDAf | float) -> NDAf | float:
    return cast("NDAf | float", 0.75 * (cosmo.Omega_b0 / cosmo.Omega_gamma0) / (1 + z))


def baumann_transfer_function(
    cosmo: StandardCosmology[NDAf, Any],
    kmag: Ann[NDAf, Doc("k magnitude [Mpc-1]")],
    z: Ann[NDAf | float, Doc("Redshift at which to evalate.")] = 1_100.0,
    /,
    *,
    z_eq: Ann[NDAf | float, Doc("Redshift at matter-radiation equality.")],
) -> NDAf | float:
    """Transfer function from Baumann."""
    # Baumann book Section C
    keq2 = (
        (cosmo.H0 / cosmo.constants.c) ** 2 * 2 * cosmo.Omega_m0 * (1 + z_eq)
    )  # C:15 [Mpc-2]
    Req = _ratio(cosmo, z_eq)  # C:17
    R = _ratio(cosmo, z)  # C:17
    rs = (  # C:16  [Mpc]
        (2 / 3)
        * np.sqrt(6 / Req / keq2)  # this is where rs gets the [Mpc]
        * np.log((np.sqrt(1 + R) + np.sqrt(R + Req)) / (1 + np.sqrt(Req)))
    )
    return cast(  # Baumann book: 7.112
        "NDAf | float", 0.2 * (np.cos(rs * kmag) / np.power(1 + R, 1 / 4) - 3 * R)
    )


def sachs_wolfe_damping(
    kmag: Ann[NDAf, Doc("Magnitude of :math:`k`. Units of [Mpc-1].")],
    /,
    *,
    kdamping: Ann[NDAf | float, Doc("Scattering damping scale. Units of [Mpc-1].")],
) -> NDAf:
    """Sachs-Wolfe damping."""
    return cast("NDAf", np.exp(-2 * (kmag / kdamping) ** 2))


# -----------------------------------------------------------------------------


def _correlation_function_integrand(  # noqa: PLR0913
    kmag: NDAf,
    z1: NDAf,
    z2: NDAf,
    dr: NDAf,
    /,
    cosmo: StandardCosmology[NDAf, Any],
    *,
    z_eq: NDAf | float,
    kdamping: NDAf | float,
    pivot_scale: NDAf | float,
    ns: NDAf | float,
) -> Ann[NDAf, Doc("(R, K) [Mpc]")]:
    # The return unit is [Mpc] because the integrand is multiplied by dk.

    prefactor = (  # [Mpc-1]
        2  # from the e^ikx integral
        * 2
        * np.pi  # from the integral dOmega
        * kmag  # from the dcostheta
    )
    # We actually include the 1/dr factor here as it keeps the integrand
    # well regulated. We then do not need to divide by dr in the end.
    return cast(
        "NDAf",
        prefactor
        * (
            power_spectrum(kmag, pivot_scale=pivot_scale, ns=ns)  # (1, K) [Mpc3]
            * baumann_transfer_function(cosmo, kmag, z1, z_eq=z_eq)  # (1, K)
            * baumann_transfer_function(cosmo, kmag, z2, z_eq=z_eq)  # (1, K)
            * sachs_wolfe_damping(kmag, kdamping=kdamping)  # (1, K)
            * np.sin(kmag * dr)  # (R, K)
            / dr  # (R, 1)  [Mpc-1]
        ),
    )


def correlation_function(  # noqa: PLR0913
    cosmo: StandardCosmology,
    kmag: Ann[NDAf, Doc("Shape (K,). Units of [Mpc-1].")],
    drs: Ann[NDAf, Doc("Shape (R,). Units of [Mpc].")],
    /,
    z1: Ann[NDAf, Doc("Redshift of scattering 1.")],
    z2: Ann[NDAf, Doc("Redshift of scattering 2.")],
    *,
    pivot_scale: Ann[NDAf, Doc("Pivot scale. [Mpc-1].")],
    ns: Ann[NDAf, Doc("Scalar spectral index.")],
    z_eq: Ann[NDAf, Doc("Redshift of matter-radiation equality.")],
    kdamping: Ann[NDAf | float, Doc("Scattering damping scale. [Mpc-1].")],
    ius_kw: dict[str, Any] | None = None,
) -> Ann[NDAf, Doc("Shape (R,). Units of [K^2].")]:
    """Correlation function (divided by $A_s$)."""
    kmag = kmag[None, :]  # (1, K) [Mpc-1]
    drs = drs[:, None]  # (R, 1) [Mpc]

    # We actually include the 1/dr factor here as it keeps the integrand
    # well regulated. We then do not need to divide by dr in the end.
    cfi = _correlation_function_integrand(  # (R, K) [Mpc]
        kmag,
        z1,
        z2,
        drs,
        cosmo=cosmo,
        z_eq=z_eq,
        kdamping=kdamping,
        pivot_scale=pivot_scale,
        ns=ns,
    )

    kw = {"k": 3, "ext": 2} if ius_kw is None else ius_kw
    integral = np.zeros(drs.size)  # (R,)
    for i in range(drs.size):
        spl = InterpolatedUnivariateSplinewithUnits(kmag, cfi[i, :], **kw)  # (K,)
        integral[i] = spl.integral(kmag.min(), kmag.max())  # (1,)

    return cast("NDAf", cosmo.T_cmb0**2 * integral)
