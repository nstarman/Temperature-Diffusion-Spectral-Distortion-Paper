"""Spectral Distortion."""

from __future__ import annotations

__all__: list[str] = []

from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from scipy.interpolate import CubicSpline

    from ..typing import NDAf


def z_of_rho(rho: NDAf, /, *, z_eq: float) -> NDAf:
    r""":math:`z = \sqrt{(s_{\|}+rho-rho_V)^2 + s_{\perp}^2}`.

    Parameters
    ----------
    rho : float
        Rho.
    z_eq : float
        Redshift at matter-radiation equality.

    Returns
    -------
    float

    """
    return (1 + z_eq) / (2 * rho**2 - 1) - 1


def rho_of_z(z: NDAf, /, *, z_eq: float) -> NDAf:
    """Rho from redshift."""
    if np.any(z < 0):
        msg = "z must be greater than 0"
        raise ValueError(msg)
    return cast("NDAf", np.sqrt((1 + (1 + z_eq) / (1 + z)) / 2))


def rho2_of_rho1(
    rho1: NDAf, /, spll: NDAf | float, sprp: NDAf | float, *, maxrho: float
) -> NDAf:
    r""":math:`rho_2 = rho_1 - \sqrt{(s_{\|}+rho_1-rho_V)^2 + s_{\perp}^2}`.

    Parameters
    ----------
    rho1 : float
        Rho.
    spll, sprp : float
        S.
    maxrho : float
        Maximum valid rho.

    Returns
    -------
    float

    """
    return cast("NDAf", rho1 - np.sqrt((spll + rho1 - maxrho) ** 2 + sprp**2))


def cubic_global_coeffs_from_ppoly_coeffs(
    spl: CubicSpline,
) -> tuple[float, float, float, float]:
    """Convert PPoly coefficients to global coefficients.

    ::
        c3(x-xi)^3 + c2(x-xi)^2 + c1(x-xi) + c0
        = p3 x^2 + p2 x^2 + p1 x + p0.

        p3 = c3
        p2 = -3 c3 xi + c2
        p1 = 3 c3 xi^2 - 2 c2 xi + c1
        p0 = -c3 xi^3 + c2 xi^2 - c1 xi + c0
    """
    xi = spl.x[:-1]
    c3, c2, c1, c0 = spl.c

    p3 = c3
    p2 = -3 * c3 * xi + c2
    p1 = 3 * c3 * xi**2 - 2 * c2 * xi + c1
    p0 = -c3 * xi**3 + c2 * xi**2 - c1 * xi + c0
    return p3, p2, p1, p0
