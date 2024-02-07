"""Simulate the P2LS data on a grid of spll, sprp."""

from __future__ import annotations

import sys
from itertools import pairwise
from typing import TYPE_CHECKING, cast
from typing import Annotated as Ann

import numpy as np
from interpolated_coordinates.utils import InterpolatedUnivariateSplinewithUnits
from showyourwork.paths import user as user_paths
from tqdm import tqdm

paths = user_paths()

sys.path.append(paths.scripts.parent.as_posix())
# isort: split

from scripts.src.cf import correlation_function

if TYPE_CHECKING:
    import astropy.units as u
    from cosmology.api import StandardCosmology
    from scipy.interpolate import InterpolatedUnivariateSpline
    from typing_extensions import Doc

    from scripts.src.typing import NDAf


##############################################################################


def get_subsamples(samples: NDAf, rng: np.random.Generator) -> tuple[NDAf, NDAf, NDAf]:
    """Get subsamples of the samples."""
    # Getting random indices for Monte Carlo integration
    idxn1a, idxn2a, idxn2b = rng.integers(0, len(samples), size=(3, 10_000_000))
    # Cleaning the indices
    repeats = (idxn1a == idxn2a) | (idxn1a == idxn2b) | (idxn2a == idxn2b)
    idxn1a = idxn1a[~repeats]
    idxn2a = idxn2a[~repeats]
    idxn2b = idxn2b[~repeats]
    # Getting the samples
    sn1a = samples[idxn1a]
    sn2a = samples[idxn2a]
    sn2b = samples[idxn2b]

    return sn1a, sn2a, sn2b


def r_distance_between_s_at_los(
    s1: Ann[NDAf, Doc(r"(N, 3) $s_{||}, \s_\perp, \phi$")],
    s2: Ann[NDAf, Doc(r"(N, 3) $s_{||}, \s_\perp, \phi$")],
    chi: Ann[NDAf, Doc("Angle [rad] between lines of sight")],
    *,
    r0: Ann[NDAf | float, Doc("Distance to the source [Mpc]")],
    Leq: Ann[NDAf | float, Doc("Distance scale factor [Mpc]")],
) -> Ann[NDAf, Doc("[Mpc]")]:
    r"""Cylindrical distance |s1 - s2| in the $s_{||}, \s_\perp, \phi$ space."""
    r0_l0 = r0 / Leq  # [none]
    termx: NDAf = (
        (r0_l0 + s2[:, 0, :]) * np.sin(chi)
        + s2[:, 1] * np.cos(chi) * np.cos(s2[:, 2])
        - s1[:, 1] * np.cos(s1[:, 2])
    )
    termy = -s2[:, 1] * np.sin(s2[:, 2]) + s1[:, 1] * np.sin(s1[:, 2])
    termz: NDAf = (
        r0_l0 * (np.cos(chi) - 1)
        + s2[:, 0] * np.cos(chi)
        - s1[:, 0]
        - s2[:, 1] * np.sin(chi) * np.cos(s2[:, 2])
    )
    return cast("NDAf", Leq * np.sqrt(termx**2 + termy**2 + termz**2))


def lines_of_sight_correlation(  # noqa: PLR0913
    cosmo: StandardCosmology,
    xispl: Ann[InterpolatedUnivariateSpline, Doc("[Mpc] -> [K^2]")],
    sn1a: Ann[NDAf, Doc("(N, 3, 1)")],
    sn2a: Ann[NDAf, Doc("(N, 3, 1)")],
    sn2b: Ann[NDAf, Doc("(N, 3, 1)")],
    /,
    chi: Ann[NDAf, Doc("(1, C), Angle [rad] between lines of sight")],
    *,
    r0: Ann[u.Quantity, Doc("distance [Mpc]")],
    Leq: Ann[NDAf | float, Doc("relevant scale [Mpc]")],
) -> Ann[NDAf, Doc("[none]")]:
    """Calculate correlation integral using Monte Carlo integration."""
    # Cross-terms depend on chi2  # (N, C) [Mpc]
    r1a2a = r_distance_between_s_at_los(sn1a, sn2a, chi=chi, r0=r0, Leq=Leq)
    r1a2b = r_distance_between_s_at_los(sn1a, sn2b, chi=chi, r0=r0, Leq=Leq)

    # Number of points
    n_pnt = len(sn1a)

    # Terms
    bbox = xispl._data[3:5] * xispl.x_unit  # noqa: SLF001
    outofbounds = (
        ((r1a2a < bbox[0]) | (r1a2a > bbox[1]) | (r1a2b < bbox[0]) | (r1a2b > bbox[1]))
        .sum(0)  # if any Out of Bounds (OoB) at a chi
        .astype(bool)
    )

    # Calculate integral (N, C) -> (1, C)
    crossterm = np.sum(xispl(r1a2a) * xispl(r1a2b), 0) / n_pnt
    crossterm[outofbounds] = np.nan  # NaN out any with an OoB

    selfterm = (np.sum(xispl(r1a2a), 0) / n_pnt) ** 2
    selfterm[outofbounds] = np.nan

    return cast("NDAf", (crossterm - selfterm) / cosmo.T_cmb0**4)


def calculate_correlation_z1z2(  # noqa: PLR0913
    z1: NDAf,
    z2: NDAf,
    chis: Ann[NDAf, Doc("Angle [rad] between lines of sight")],
    /,
    cosmo: StandardCosmology,
    samples: NDAf,
    ks: NDAf,
    drs: NDAf,
    *,
    Leq: u.Quantity,
    r0: u.Quantity,
    ns: NDAf,
    z_eq: NDAf,
    kdamping: u.Quantity,
    pivot_scale: u.Quantity,
    rng: np.random.Generator,
    n_chi_steps: int = 1,
) -> NDAf:
    """Calculate the correlation function between z1 and z2 along separations chi."""
    # Get samples
    sn1a, sn2a, sn2b = get_subsamples(samples, rng)

    # Calculate the correlation function (divded by A_s) [K^2]
    xi = correlation_function(
        cosmo,
        ks,
        drs,
        z1=z1,
        z2=z2,
        pivot_scale=pivot_scale,
        ns=ns,
        z_eq=z_eq,
        kdamping=kdamping,
        ius_kw={"k": 3, "ext": 2},
    )
    # Splining for later integration [Mpc] -> [K^2]
    xispl = InterpolatedUnivariateSplinewithUnits(drs, xi, k=3, ext=1)

    out = np.empty(len(chis))
    for i, j in tqdm(tuple(pairwise(range(0, len(chis) + n_chi_steps, n_chi_steps)))):
        out[i:j] = lines_of_sight_correlation(
            cosmo,
            xispl,
            sn1a[..., None],
            sn2a[..., None],
            sn2b[..., None],
            chi=chis[None, i:j],
            r0=r0,
            Leq=Leq,
        )
    return out
