"""Convert between different distance measures."""

from __future__ import annotations

__all__: list[str] = []

from typing import TYPE_CHECKING, cast
from typing import Annotated as Ann

import numpy as np

if TYPE_CHECKING:
    from typing_extensions import Doc

    from .qclassy import StandardCosmologyWrapper
    from .typing import NDAf


def Leq(
    cosmo: StandardCosmologyWrapper,
    *,
    z_eq: Ann[NDAf | float, Doc("Redshift at matter-radiation equality.")],
) -> Ann[NDAf, Doc("Distance scale factor, [Mpc]")]:
    """Distance scale factor."""
    return cast(
        "NDAf",
        (cosmo.constants.c / cosmo.H0)
        * np.sqrt(8 * cosmo.scale_factor(z_eq) / cosmo.Omega_m0),
    )


def z_recombination(cosmo: StandardCosmologyWrapper, /) -> NDAf:
    """Compute redshift of recombination.

    Uses the CLASSY thermodynamics module to find the redshift of
    the maximum of the visibility function.
    """
    th = cosmo.cosmo.get_thermodynamics()
    return np.array(th["z"][th["g [Mpc^-1]"].argmax()])


##############################################################################


def z_of_s(s: NDAf | float, /, *, z_eq: float, z0: float) -> NDAf | float:
    """Redshift from s."""
    return cast(
        "NDAf",
        (1 + z_eq) / ((np.sqrt(1 + (1 + z_eq) / (1 + z0)) - np.sqrt(2) * s) ** 2 - 1)
        - 1,
    )


def s_of_z(z: NDAf | float, /, *, z_eq: float, z0: float) -> NDAf | float:
    """Return s from redshift."""
    if z < z0:
        msg = "z must be greater than z0"
        raise ValueError(msg)

    return cast(
        "NDAf",
        (np.sqrt(1 + (1 + z_eq) / (1 + z0)) - np.sqrt(1 + (1 + z_eq) / (1 + z)))
        / np.sqrt(2),
    )


def r_distance_between_s(
    s1: Ann[NDAf, Doc(r"(N, 3) array $s_{||}, \s_\perp, \phi$")],
    s2: Ann[NDAf, Doc(r"(N, 3) array $s_{||}, \s_\perp, \phi$")],
    *,
    Leq: Ann[NDAf | float, Doc("Distance scale factor [Mpc]")],
) -> Ann[NDAf, Doc(r"(N,) array of distances [Mpc]")]:
    r"""Cylindrical distance |s1 - s2| in the $s_{||}, \s_\perp, \phi$ space.

    .. math::

        d^2 = s1_\perp^2 + s2_\perp^2 - 2 s1_\perp s2_\perp \cos(\phi_1 - \phi_2) + (s1_{||} - s2_{||})^2
    """  # noqa: E501
    return cast(
        "NDAf",
        Leq
        * np.sqrt(
            s1[:, 1] ** 2
            + s2[:, 1] ** 2
            - 2 * s1[:, 1] * s2[:, 1] * np.cos(s1[:, 2] - s2[:, 2])
            + (s1[:, 0] - s2[:, 0]) ** 2
        ),
    )
