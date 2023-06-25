"""Spectral Distortion."""

from __future__ import annotations

__all__: list[str] = []

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias
from typing import Annotated as Ann

import astropy.units as u
import numpy as np
import tqdm
from scipy.interpolate import (
    CubicSpline,
    InterpolatedUnivariateSpline,
    RectBivariateSpline,
)

from ..distances import Leq as compute_Leq
from ..prob_2ls.utils import (
    cubic_global_coeffs_from_ppoly_coeffs,
    rho2_of_rho1,
    rho_of_z,
    z_of_rho,
)

if TYPE_CHECKING:
    from cosmology.compat.classy import StandardCosmologyWrapper
    from typing_extensions import Doc

    from ..typing import NDAf


ArgsT: TypeAlias = tuple[
    "NDAf", "NDAf", "NDAf", float, float, "NDAf", "NDAf", "NDAf", "NDAf", "NDAf", "NDAf"
]


@dataclass(frozen=True)
class ComputePspllSprp:
    """Compute P(spll, sprp) for a given spectral distortion."""

    z_eq: float  # [dimensionless]
    Leq: float  # [Mpc]
    rho_domain: tuple[float, float]  # (min, max)

    spl_ln_gbarCL: InterpolatedUnivariateSpline  # ([dimensionless]) -> [Mpc]
    spl_ln_PbarCL: InterpolatedUnivariateSpline  # ([dimensionless]) -> [Mpc]

    @classmethod
    def from_CLASS(
        cls,
        cosmo: StandardCosmologyWrapper,
        /,
        z_eq: float,
        z_bounds: tuple[float, float] = (5500, 100),
    ) -> ComputePspllSprp:
        """Initialize from a given format."""
        Leq = (compute_Leq(cosmo, z_eq=z_eq) << u.Mpc).value  # [Mpc]

        # --- thermodynamic quantities --- #

        th = cosmo.cosmo.get_thermodynamics()

        _in_bounds = (min(z_bounds) <= th["z"]) & (th["z"] <= max(z_bounds))
        z = th["z"][_in_bounds][::-1]
        z.flags.writeable = False

        g = th["g [Mpc^-1]"][_in_bounds][::-1].copy()  # z-ordered [Mpc^-1]
        g.flags.writeable = True

        # Add derived values
        rho = rho_of_z(z, z_eq=z_eq)
        rho_domain = (rho[0], rho[-1])

        # --- splines --- #

        # ext 1 makes stuff 0, which is correct, but introduces a small discontinuity.
        spl_gbarCL = InterpolatedUnivariateSpline(rho, g, ext=1)  # [Mpc^-1]

        # For the log we need to avoid log(0), and also avoid a discontinuity,
        # and ensure that the extrapolation is reasonable. Ext 3 returns the boundary
        # value, which is what we want.
        g = g.copy()
        g[g <= 0] = 1e-300  # should be small enough that discontinuity is negligible
        spl_ln_gbarCL = InterpolatedUnivariateSpline(rho, np.log(g), ext=3)

        # Instead, to get the right normalization we will define PbarCL from an integral
        # over gbarCL.  # [Mpc^-1]
        integral = [spl_gbarCL.integral(a, b) for a, b in itertools.pairwise(rho)]
        PbarCL = Leq * np.concatenate(([0], np.cumsum(integral)))  # [none]
        _spl_PbarCL = InterpolatedUnivariateSpline(rho, PbarCL, ext=2)

        # normalize the spline
        a, b = rho_domain  # [none]
        norm = Leq * spl_gbarCL.integral(a, b) / _spl_PbarCL(b)  # [none]
        norm_PbarCL = PbarCL / norm  # [none]

        # For the log we need to avoid log(0) and also avoid a discontinuity
        # then we fix by extrapolating from the adjacent two points.
        norm_PbarCL[0] = norm_PbarCL[1]  # known to be 0, avoid log(0)
        lnPbarCL = np.log(norm_PbarCL)
        lnPbarCL[0] = lnPbarCL[1] - (lnPbarCL[2] - lnPbarCL[1]) / 2
        spl_ln_PbarCL = InterpolatedUnivariateSpline(rho, lnPbarCL, ext=2)

        return cls(
            z_eq=z_eq,
            Leq=Leq,
            rho_domain=rho_domain,
            spl_ln_gbarCL=spl_ln_gbarCL,
            spl_ln_PbarCL=spl_ln_PbarCL,
        )

    # ------------------------------------

    @property
    def prefactor(self) -> NDAf:
        """The prefactor for the integral."""
        return np.asarray(
            3 * self.Leq**2 / (16 * np.exp(self.spl_ln_PbarCL(self.rho_domain[-1])))
        )

    # ------------------------------------
    # Integrals between knots (vectorized)

    def _integral0(self, args: ArgsT, /) -> NDAf:
        xi, xj, _, pllrO, sprp, _, _, xjs2, xis2, diffatan, _ = args
        return (
            -sprp * ((xj + pllrO) / xjs2 - (xi + pllrO) / xis2)  # delta 1st term
            + 3 * diffatan  # delta 2nd term
        )

    def _integral1(self, args: ArgsT, /) -> NDAf:
        pllrO, sprp, xjpllrO, xipllrO, xjs2, xis2, diffatan, difflog = args[3:]
        t1: NDAf = (
            +(sprp * pllrO * xjpllrO + sprp**3) / xjs2
            - (sprp * pllrO * xipllrO + sprp**3) / xis2
            - 3 * pllrO * diffatan
            + 2 * sprp * difflog
        )
        return t1

    def _integral2(self, args: ArgsT, /) -> NDAf:
        xi, xj, dx, pllrO, sprp, xjpllrO, xipllrO, xjs2, xis2, diffatan, difflog = args
        t2: NDAf = (
            +5 * sprp * dx
            - sprp * xjpllrO * xj**2 / xjs2
            + sprp * xipllrO * xi**2 / xis2
            + (3 * pllrO**2 - 5 * sprp**2) * diffatan
            - 4 * pllrO * sprp * difflog
        )
        return t2

    def _integral3(self, args: ArgsT, /) -> NDAf:
        xi, xj, dx, pllrO, sprp, xjpllrO, xipllrO, xjs2, xis2, diffatan, difflog = args
        t3: NDAf = (
            +3 * sprp * ((xj**2 - xi**2) - 3 * pllrO * dx)
            - sprp * xjpllrO * xj**3 / xjs2
            + sprp * xipllrO * xi**3 / xis2
            - 3 * pllrO * (pllrO**2 - 5 * sprp**2) * diffatan
            + 6 * sprp * (pllrO**2 - 0.5 * sprp**2) * difflog
        )
        return t3

    # ------------------------------------

    def _visibilities_factor(self, rho: NDAf, spll: float, sprp: float) -> NDAf:
        """Return log [g/P](rho_1)*g(rho2)."""
        rho2 = rho2_of_rho1(rho, spll, sprp, maxrho=self.rho_domain[-1])
        vf: NDAf = np.exp(
            self.spl_ln_gbarCL(rho) - self.spl_ln_PbarCL(rho) + self.spl_ln_gbarCL(rho2)
        )
        return vf

    def _make_visibilities_spline(
        self, rho: NDAf, spll: float, sprp: float
    ) -> CubicSpline:
        gs = self._visibilities_factor(rho, spll, sprp)
        gs[gs < 0] = 0  # shouldn't happen
        gs[np.isnan(gs)] = 0  # shouldn't happen
        return CubicSpline(rho, gs)

    def _setup(self, rho: NDAf, spll: float, sprp: float) -> ArgsT:
        xi = rho[:-1]  # (N-1,)
        xj = rho[1:]  # (N-1,)
        dx = xj - xi  # (N-1,)
        pllrO = spll - self.rho_domain[-1]
        xipllrO = xi + pllrO
        xjpllrO = xj + pllrO
        xis2 = xipllrO**2 + sprp**2
        xjs2 = xjpllrO**2 + sprp**2

        # Trick for difference of arctans with same denominator
        # beta - alpha = arctan( (tan(beta)-tan(alpha)) / (1 + tan(alpha)tan(beta)) )
        #              = arctan( c(b - a) / (c^2 + a*b ) )
        diffatan = (np.arctan2(sprp * dx, sprp**2 + xipllrO * xjpllrO) << u.rad).value

        # Log term
        difflog = np.log(xjs2 / xis2)

        return xi, xj, dx, pllrO, sprp, xjpllrO, xipllrO, xjs2, xis2, diffatan, difflog

    def __call__(
        self, z: Ann[NDAf, Doc("Redshift.")], spll: float, sprp: float
    ) -> NDAf:
        r"""Evaluate :math:`\mathcal{P}(s_{||}, s_\perp)`."""
        rho = rho_of_z(z, z_eq=self.z_eq)
        spl = self._make_visibilities_spline(rho, spll, sprp)  # [Mpc-1]
        p3, p2, p1, p0 = cubic_global_coeffs_from_ppoly_coeffs(spl)

        args = self._setup(rho, spll, sprp)

        t0 = self._integral0(args)
        t1 = self._integral1(args)
        t2 = self._integral2(args)
        t3 = self._integral3(args)

        P: NDAf = self.prefactor * np.sum(p0 * t0 + p1 * t1 + p2 * t2 + p3 * t3)
        return P


# =============================================================================


def _make_z(  # noqa: PLR0913
    Pfunc: ComputePspllSprp,
    /,
    spll: float,
    sprp: float,
    *,
    n_sprp: Ann[int, Doc("Number of grid points in the s_perp direction")] = 15,
    n_z_center: Ann[int, Doc("Number of rho in the center array")] = 1_000,
    n_z_lr: Ann[int, Doc("Number of rho in left and right arrays.")] = 1_000,
) -> NDAf:
    """Make z array for integration."""
    # Center region
    center = Pfunc.rho_domain[-1] - spll
    center_lower = max(center - n_sprp * np.abs(sprp), Pfunc.rho_domain[0])
    center_upper = min(center + n_sprp * np.abs(sprp), Pfunc.rho_domain[1])

    # Add lower rho, if center doesn't hit lower bound
    rho_lower: NDAf
    if center_lower == Pfunc.rho_domain[0]:
        rho_lower = np.array([])
    else:
        rho_lower = np.linspace(Pfunc.rho_domain[0] + 1e-5, center_lower, num=n_z_lr)

    rho_center = np.linspace(center_lower, center_upper, num=n_z_center)

    # Add upper rho, if center doesn't hit upper bound
    rho_upper: NDAf
    if center_upper == Pfunc.rho_domain[1]:
        rho_upper = np.array([])
    else:
        rho_upper = np.linspace(center_upper, Pfunc.rho_domain[1], num=n_z_lr)

    # TODO: not need the `unique` check
    rho: NDAf = np.unique(np.concatenate((rho_lower[:-1], rho_center, rho_upper[1:])))
    return z_of_rho(rho, z_eq=Pfunc.z_eq)


def compute_on_grid(  # noqa: PLR0913
    Pfunc: ComputePspllSprp,
    Spll: Ann[NDAf, Doc(r"$s_{||}$. Columnar")],
    Sprp: Ann[NDAf, Doc(r"$s_\perp$. Row-wise")],
    *,
    n_sprp: Ann[int, Doc("Number of grid points in the s_perp direction")] = 15,
    n_z_center: Ann[int, Doc("Number of rho in the center array")] = 1_000,
    n_z_lr: Ann[int, Doc("Number of rho in left and right arrays.")] = 1_000,
) -> Ann[tuple[NDAf, float], Doc("value, correction")]:
    """Compute on a grid."""
    Parr = np.zeros_like(Spll)
    shape = Parr.shape

    z_kw = {"n_sprp": n_sprp, "n_z_center": n_z_center, "n_z_lr": n_z_lr}
    for i, j in tqdm.tqdm(
        itertools.product(range(shape[0]), range(shape[1])),
        total=shape[0] * shape[1],
    ):
        z = _make_z(Pfunc, Spll[i, j], Sprp[i, j], **z_kw)
        Parr[i, j] = Pfunc(z, Spll[i, j], Sprp[i, j])

    _spl = RectBivariateSpline(Spll[:, 0], Sprp[0, :], Parr, kx=3, ky=3, s=0)
    # the correction should be 1.
    correction = _spl.integral(Spll.min(), Spll.max(), Sprp.min(), Sprp.max())

    return Parr / correction, correction
