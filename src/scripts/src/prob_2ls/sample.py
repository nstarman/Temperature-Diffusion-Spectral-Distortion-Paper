r"""Draw representative samples from :math:`\mcal{P}(s_{||}, s_{\perp}, \phi)`."""

from __future__ import annotations

__all__: list[str] = []

import copy as pycopy
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, TypeVar, cast

import numpy as np
from numpy.random import Generator, default_rng
from scipy.interpolate import (
    CubicSpline,
    InterpolatedUnivariateSpline,
    LinearNDInterpolator,
    RectBivariateSpline,
)
from scipy.stats import rv_continuous

if TYPE_CHECKING:
    from typing import Any

    from ..typing import NDAf


##############################################################################


Self_Pspll = TypeVar("Self_Pspll", bound="Pspll_Distribution")


class Pspll_Distribution(rv_continuous):  # type: ignore[misc]
    r"""Distribution of :math:`\mathcal{P}(s_{||})`.

    Parameters
    ----------
    P_spll_spl : `~scipy.interpolate.CubicSpline`
        Spline of :math:`\mathcal{P}(s_{||},s_{\perp})`.
    a, b : float, optional
        Lower/upper bound of the support.
    xtol : float, optional
        Tolerance for the support.
    badvalue : Any, optional
        Value to return for out-of-support points.
    name : str, optional
        Name of the distribution.
    longname : str, optional
        Long name of the distribution.
    shapes : tuple, optional
        Shape parameters.
    seed : int, optional
        Seed for the random number generator.

    """

    def __init__(  # noqa: PLR0913
        self,
        P_spll_spl: CubicSpline,
        a: float | None = None,
        b: float | None = None,
        xtol: float = 1e-14,
        badvalue: Any | None = None,
        name: str | None = None,
        longname: str | None = None,
        shapes: tuple[int, ...] | None = None,
        seed: int | None = None,
        *,
        cdf: CubicSpline | None = None,
    ) -> None:
        # PDF
        self._spl_pdf = pycopy.deepcopy(P_spll_spl)

        # The CDF
        cdf_spl = cdf if cdf is not None else P_spll_spl.antiderivative()

        # We need to ensure the normalization of the CDF
        cdf_arr = cdf_spl(self.x)
        cdfmax = cdf_arr.max()
        if not np.allclose(cdfmax, 1):
            msg = "CDF is not normalized"
            raise ValueError(msg)
        # still might need a bit of normalization
        elif cdfmax < 1:  # noqa: RET506
            cdf_arr /= cdfmax
            self._spl_cdf = CubicSpline(self.x, cdf_arr, extrapolate=False)
        else:
            self._spl_cdf = cdf_spl

        # The PPF, the most important for sampling
        # To generate the PPF, we need to invert the CDF, so we super-sample the CDF
        # and then numerically invert it.
        _y = np.linspace(self.x[0], self.x[-1], 10 * len(self.x), endpoint=True)
        cdf_arr = self._spl_cdf(_y)

        # We need to ensure that the CDF is strictly increasing
        # This is not always the case because of numerical errors
        # So we need to truncate the CDF where it is not strictly increasing.
        # TODO: an error check here would be nice
        upto = np.where(np.diff(cdf_arr) <= 0)[0]
        if len(upto) > 0:
            to: int = upto[0]
            _y = _y[:to]
            cdf_arr = cdf_arr[:to]
        self._spl_ppf = InterpolatedUnivariateSpline(cdf_arr, _y, ext=2)

        # Bounds
        if a is None:
            a = float(P_spll_spl.x[0])
        if b is None:
            b = float(P_spll_spl.x[-1])

        super().__init__(
            momtype=1,
            a=a,
            b=b,
            xtol=xtol,
            badvalue=badvalue,
            name=name,
            longname=longname,
            shapes=shapes,
            seed=seed,
        )

    # The PDF, CDF, and PPF are all defined in terms of the spline
    def _pdf(self, x: NDAf) -> NDAf:
        return cast("NDAf", self._spl_pdf(x))

    def _cdf(self, x: NDAf) -> NDAf:
        return cast("NDAf", self._spl_cdf(x))

    def _ppf(self, q: NDAf) -> NDAf:
        return cast("NDAf", self._spl_ppf(q))

    @property
    def x(self) -> NDAf:
        """Support of the distribution."""
        return cast("NDAf", self._spl_pdf.x)

    # -------------------------------------------------------------------------

    @classmethod
    def from_Pspllsprp(
        cls: type[Self_Pspll],
        spll: NDAf,
        Pspllsprp: RectBivariateSpline,
        *args: Any,
        b: float | None = None,
        **kwargs: Any,
    ) -> Self_Pspll:
        r"""Construct :math:`P(s_{||})` from :math:`P(s_{||}, s_{\perp})`.

        Parameters
        ----------
        spll, sprp : NDAf
            Support of :math:`s_{||}, s_{\perp}`.
        Pspllsprp : `~scipy.interpolate.RectBivariateSpline`
            Distribution of :math:`\mathcal{P}(s_{||},s_{\perp})`, evaluated at the
            points 'spll', 'sprp'.
        *args, **kwargs: Any
            Arguments passed to the constructor.

        b : float, optional keyword-only
            Upper bound of the support. :math:`\mathcal{P}(s_{||},s_{\perp})` is 0 for
            :math:`s_{||} > b`, which is physically true, but might not be reflected in
            the numerics.

        Returns
        -------
        Pspll_Distribution
            Distribution of :math:`\mathcal{P}(s_{||})`.

        """
        # Cut the support to `b`
        spll = spll[spll <= b] if b is not None else spll
        # We neeed the midpoints of
        spllmid = (spll[:-1] + spll[1:]) / 2

        # Marginalize over s_perp
        _P_spll = np.zeros(len(spllmid), dtype=float)
        for i, (splli, spllj) in enumerate(pairwise(spll)):
            _P_spll[i] = np.maximum(
                0,
                Pspllsprp.integral(splli, spllj, 0, 3) / (spllj - splli),
            )
        P_spll_arr = np.array(_P_spll)

        # Spline and normalize so that the integral is 1
        P_spll_unnorm = CubicSpline(spllmid, P_spll_arr, extrapolate=False)
        norm = P_spll_unnorm.integrate(spllmid[0], spllmid[-1])
        P_spll_arr /= norm
        P_spll = CubicSpline(spllmid, P_spll_arr, extrapolate=False)

        # CDF
        cdf_spl = P_spll.antiderivative()
        cdf_arr = cdf_spl(P_spll.x)
        diffs = np.diff(cdf_arr)

        # We need to ensure that the CDF is strictly increasing, or at least has only
        # one zero derivative, where it becomes constant due to limited precision.
        if all(diffs > 0):
            pass
        elif any(diffs < 0):
            msg = "CDF is not monotonically increasing"
            raise ValueError(msg)
        elif np.diff(diffs == 0).sum() > 1:
            msg = "CDF has more than one zero derivative"
            raise ValueError(msg)
        else:  # because of limited precision, the CDF becomes constant.
            i = np.where(diffs == 0)[0][0]
            spllmid = spllmid[: i + 1]
            P_spll_arr = P_spll_arr[: i + 1]

            P_spll = CubicSpline(spllmid, P_spll_arr, extrapolate=False)
            cdf_spl = CubicSpline(spllmid, cdf_arr[: i + 1], extrapolate=False)

        return cls(P_spll, *args, cdf=cdf_spl, **kwargs)


# ==============================================================================


Self_Psprp = TypeVar("Self_Psprp", bound="Psprp_ConditionalSampler")


@dataclass(frozen=True)
class Psprp_ConditionalSampler:
    r"""Conditional distribution :math:`P(s_{\perp}|s_{||})`.

    Parameters
    ----------
    spl_pdf : `scipy.interpolate.RectBivariateSpline`
        The pdf of sprp given spll.
    spl_cdf : `scipy.interpolate.RectBivariateSpline`
        The cdf of sprp given spll.
    spl_ppf : `scipy.interpolate.LinearNDInterpolator`
        The ppf of sprp given spll.
        This is used to sample for sprp.

    """

    spl_pdf: RectBivariateSpline
    spl_cdf: RectBivariateSpline
    spl_ppf: LinearNDInterpolator

    # ==============================================================================
    # Private methods -- following the scipy.stats.rv_continuous API

    def _pdf(self, spll: NDAf, sprp: NDAf, *, grid: bool = False) -> NDAf:
        return cast("NDAf", self.spl_pdf(spll, sprp, grid=grid))

    def pdf(self, spll: NDAf, sprp: NDAf, *, grid: bool = False) -> NDAf:
        """Evaluate the pdf in sprp, given spll.

        Parameters
        ----------
        spll : array_like
            The spll values.
        sprp : array_like
            The sprp values.

        grid : bool, optional keyword-only
            If True, return the pdf on a grid of spll and sprp values.

        Returns
        -------
        array_like
            The pdf in sprp, given spll.

        """
        return self._pdf(spll, sprp, grid=grid)

    def _cdf(self, spll: NDAf, sprp: NDAf, *, grid: bool = False) -> NDAf:
        return cast("NDAf", self.spl_cdf(spll, sprp, grid=grid))

    def cdf(self, spll: NDAf, sprp: NDAf, *, grid: bool = False) -> NDAf:
        """Evalute the cdf in sprp, given spll.

        Parameters
        ----------
        spll : NDAf
            The spll values.
        sprp : NDAf
            The sprp values.

        grid : bool, optional keyword-only
            If True, return the cdf on a grid of spll and sprp values.

        Returns
        -------
        NDArray[floating[Any]]
            The cdf in sprp, given spll.

        """
        return self._cdf(spll, sprp, grid=grid)

    def _ppf(self, spll: NDAf, q: NDAf) -> NDAf:
        return cast("NDAf", self.spl_ppf(spll, q))

    def ppf(self, spll: NDAf, q: NDAf) -> NDAf:
        """Evaluate the ppf in sprp, given spll.

        Parameters
        ----------
        spll : NDAf
            The spll values.
        q : NDAf
            The quantiles.

        Returns
        -------
        NDArray[floating[Any]]
            The ppf in sprp, given spll.

        """
        return self._ppf(spll, q)

    def _rvs(self, spll: NDAf, *, size: int | None, rng: Generator) -> NDAf:
        # Use basic inverse cdf algorithm for RV generation as default.
        return self._ppf(spll, rng.uniform(size=size))

    def rvs(
        self, spll: NDAf, *, size: int | None = None, rng: Generator | None = None
    ) -> NDAf:
        """Draw random variates.

        Parameters
        ----------
        spll : NDAf
            The spll values at which to draw the random variates.
        size : int or None, optional keyword-only
            The number of variates to draw, by default `None` = len(spll)
        rng : Generator, optional keyword-only
            The random number generator to use.
            `None` (default) uses :func:`numpy.random.default_rng`.

        Returns
        -------
        NDArray[floating[Any]]
            The random variates. Shape ``size``.

        """
        if size is None:
            size = len(spll)
        return self._rvs(np.broadcast_to(spll, size), size=size, rng=default_rng(rng))

    # -------------------------------------------------------------------------

    @classmethod
    def from_marginal(
        cls: type[Self_Psprp],
        sprp: NDAf,
        Pspllsprp: RectBivariateSpline,
        Pspll: CubicSpline,
        *,
        cutoff: float = 0,
    ) -> Self_Psprp:
        r"""Construct the distribution given the pdf and marginal.

        Parameters
        ----------
        sprp : NDAf
            The sprp values.
        Pspllsprp : RectBivariateSpline
            PDF of :math:`\mathcal{P}(s_{||}, s_{\perp})`.
        Pspll : CubicSpline
            Marginal distribution :math:`\mathcal{P}(s_{||})`.

        cutoff : float, optional keyword-only
            The cutoff for the marginal distribution, by default 0.

        Returns
        -------
        ``Psprp_ConditionalSampler``

        """
        x = Pspll.x

        # Compute the conditional pdf
        pdfe = Pspllsprp(x, sprp, grid=True)  # numerator
        pdfe[pdfe < 0] = 0
        pdfe[pdfe < cutoff] = 0  # apply the cutoff

        indicator = pdfe != 0  # where to apply the denominator
        pdfe[indicator] /= (Pspll(x)[:, None] * indicator)[indicator]  # denominators
        pdfe[pdfe < 0] = 0  # again remove any negative values

        cond_spl_pdf = RectBivariateSpline(x, sprp, pdfe, kx=3, ky=3, s=0)

        # Compute the conditional cdf
        cdf = np.zeros((len(x) - 1, len(sprp) - 1), dtype=float)
        for i, (xi, xj) in enumerate(pairwise(x)):
            cdf[i, :] = np.cumsum(
                tuple(
                    np.maximum(0, cond_spl_pdf.integral(xi, xj, yi, yj))
                    for yi, yj in pairwise(sprp)
                )
            )
            cdf[i, :] /= cdf[i, -1]  # normalize each column (s_{\perp} array)

        # The x, sprp is done at the midpoint
        xmid = (x[:-1] + x[1:]) / 2
        sprpmid = (sprp[:-1] + sprp[1:]) / 2

        cond_spl_cdf = RectBivariateSpline(xmid, sprpmid, cdf, kx=3, ky=3, s=0)

        # Compute the PPF
        X, Y = np.meshgrid(xmid, sprpmid, indexing="ij")
        pnts = np.c_[X.flat, cdf.flat]
        cond_spl_ppf = LinearNDInterpolator(pnts, Y.flat)

        return cls(cond_spl_pdf, cond_spl_cdf, cond_spl_ppf)


# ==============================================================================


Self_P2D = TypeVar("Self_P2D", bound="P2D_Distribution")


@dataclass(frozen=True)
class P2D_Distribution:
    r"""Distribution in :math:`\mathcal{P}(s_{||}, s_{\perp})`.

    Parameters
    ----------
    pdf : RectBivariateSpline
        The pdf.
    marginal_spll : ``Pspll_Distribution``
        The marginal distribution in :math:`\mathcal{P}(s_{||})`.
    conditional_sprp : ``Psprp_ConditionalSampler``
        The conditional distribution in :math:`\mathcal{P}(s_{\perp} | s_{||})`.

    """

    pdf: RectBivariateSpline
    marginal_spll: Pspll_Distribution
    conditional_sprp: Psprp_ConditionalSampler

    def rvs(self, *, size: int = 1, rng: Generator | None = None) -> NDAf:
        """Draw random variates.

        Parameters
        ----------
        size : int, optional keyword-only
            Number of random variates to draw.
        rng : Generator, optional keyword-only
            The random number generator to use.

        Returns
        -------
        NDArray[floating[Any]]

        """
        samples_pll = self.marginal_spll.rvs(size=size)
        samples_prp = self.conditional_sprp.rvs(samples_pll, size=size, rng=rng)

        return cast("NDAf", np.c_[samples_pll, samples_prp])

    # ----------------------------------------------------------

    @classmethod
    def from_Pspllsprp(  # noqa: PLR0913
        cls: type[Self_P2D],
        spll: NDAf,
        sprp: NDAf,
        pdf: NDAf | RectBivariateSpline,
        *,
        spll_b: float | None = None,
        pdf_cutoff: float = 0,
    ) -> Self_P2D:
        """Construct the distribution given the pdf.

        Parameters
        ----------
        spll : NDAf
            The spll values.
        sprp : NDAf
            The sprp values.
        pdf : RectBivariateSpline
            The pdf.

        spll_b : float | None, optional keyword-only
            The upper bound of spll in the marginal, by default `None` = max(spll).
        pdf_cutoff : float, optional keyword-only
            The cutoff for the conditional distribution, by default 0.

        Returns
        -------
        ``P2D_Distribution``

        """
        pdf_spl = (
            pycopy.deepcopy(pdf)
            if isinstance(pdf, RectBivariateSpline)
            else RectBivariateSpline(spll, sprp, pdf, kx=3, ky=3, s=0)
        )

        marginal_spll = Pspll_Distribution.from_Pspllsprp(spll, pdf_spl, b=spll_b)
        conditional_sprp = Psprp_ConditionalSampler.from_marginal(
            sprp,
            pdf_spl,
            pycopy.deepcopy(marginal_spll._spl_pdf),  # noqa: SLF001
            cutoff=pdf_cutoff,
        )

        return cls(
            pdf=pdf_spl, marginal_spll=marginal_spll, conditional_sprp=conditional_sprp
        )


@dataclass(frozen=True)
class P3D_Distribution(P2D_Distribution):
    r"""Distribution of :math:`\mathcal{P}(s_{||}, s_{\perp}, \phi)`.

    Parameters
    ----------
    pdf : RectBivariateSpline
        The pdf.
    marginal_spll : ``Pspll_Distribution``
        The marginal distribution in :math:`\mathcal{P}(s_{||})`.
    conditional_sprp : ``Psprp_ConditionalSampler``
        The conditional distribution in :math:`\mathcal{P}(s_{\perp} | s_{||})`.

    """

    def rvs(self, *, size: int = 1, rng: Generator | None = None) -> NDAf:
        """Draw random variates.

        Parameters
        ----------
        size : int, optional keyword-only
            Number of random variates to draw.
        rng : Generator | None, optional keyword-only
            The random number generator to use.

        Returns
        -------
        NDArray[floating[Any]]

        """
        sprppll = super().rvs(size=size)  # (N, 2)
        phis: NDAf = default_rng(rng).uniform(low=0, high=2 * np.pi, size=size)
        return cast("NDAf", np.c_[sprppll, phis])
